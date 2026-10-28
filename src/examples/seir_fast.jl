"""
    FastSEIR

A specialized, streaming implementation of the exact SEIRS phylodynamic
particle filter (King, Lin & Ionides 2024), written for the scalability
experiments.  Compared with the generic POMP-based filters it

- compiles the genealogy into immutable arrays shared by all particles,
- stores each particle's lineage coloring as a bitmask (struct of arrays),
- keeps only the current particle population (O(`Np`) memory),
- offers several *exact* importance-sampling proposals for lineage
  colour changes, evaluated at the actual jump time,
- can process each genealogical event *before* resampling (`order = :post`),
- optionally triggers resampling on the effective sample size,
- propagates particles in parallel with reproducible per-chunk RNG streams.

All proposals target the same filter equation, so every configuration
estimates the same likelihood (see `ExactSEIR` for the reference).
"""
module FastSEIR

using Random: AbstractRNG, Xoshiro, default_rng, randexp
using Base.Threads: @threads, nthreads
using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time, Genealogy, Guide
import PartiallyObservedMarkovProcesses as POMP

const ROOT = 0x00
const SAMPLE = 0x01
const BRANCH = 0x02

"""
    Tree(g, guide = nothing)

Compile genealogy `g` (and optionally a [`Guide`](@ref) built on it) into flat
arrays.  Interval `k` is `[time[k], tend[k])`, following event `k`.
"""
struct Tree
    t0::Float64
    T::Float64
    time::Vector{Float64}
    tend::Vector{Float64}
    type::Vector{UInt8}
    par::Vector{Int32}
    ch1::Vector{Int32}
    ch2::Vector{Int32}
    nch::Vector{Int8}
    nlin::Int
    W::Int
    ## present lineages on interval k: lins[ptr[k]:ptr[k+1]-1]
    ptr::Vector{Int}
    lins::Vector{Int32}
    ## guide: p_i(t) = A[i,1]*c1 + A[i,2]*exp(-κ*(tend-t))*c2, aligned with `lins`
    guided::Bool
    c1::Vector{Float64}
    c2::Vector{Float64}
    A::NTuple{4,Float64}        # A11, A21, A12, A22 (deme 1 = E, deme 2 = I)
    κ::Float64
    ## guide probabilities of the child lineages at the event (deme E, deme I)
    pres::Matrix{Float64}       # 4 × M: (c1 in E, c1 in I, c2 in E, c2 in I)
end

Tree(g::Genealogy, guide::Union{Nothing,Guide} = nothing) = begin
    M = length(g)
    time = [g[k].slate for k ∈ 1:M]
    tend = [k < M ? g[k+1].slate : g.time for k ∈ 1:M]
    type = UInt8[g[k].type == Root ? ROOT : g[k].type == Sample ? SAMPLE : BRANCH for k ∈ 1:M]
    par = Int32[g[k].lineage for k ∈ 1:M]
    nch = Int8[length(g[k].children) for k ∈ 1:M]
    ch1 = Int32[nch[k] ≥ 1 ? g[g[k].children[1]].lineage : 0 for k ∈ 1:M]
    ch2 = Int32[nch[k] ≥ 2 ? g[g[k].children[2]].lineage : 0 for k ∈ 1:M]
    nlin = Int(g.nsample)
    W = cld(max(nlin,1), 64)
    ptr = Vector{Int}(undef, M+1)
    lins = Int32[]
    present = Set{Int32}()
    for k ∈ 1:M
        delete!(present, par[k])
        nch[k] ≥ 1 && push!(present, ch1[k])
        nch[k] ≥ 2 && push!(present, ch2[k])
        ptr[k] = length(lins)+1
        append!(lins, sort!(collect(present)))
    end
    ptr[M+1] = length(lins)+1
    c1 = zeros(length(lins)); c2 = zeros(length(lins))
    pres = fill(0.5, 4, M)
    A = (0.0, 0.0, 0.0, 0.0); κ = 0.0
    if !isnothing(guide)
        m = guide.process
        @assert length(m.eigenvals) == 2 "FastSEIR guides must have two demes"
        A = (m.A[1,1], m.A[2,1], m.A[1,2], m.A[2,2])
        κ = -m.eigenvals[2]
        for k ∈ 1:M
            gn = guide[k]
            for j ∈ ptr[k]:(ptr[k+1]-1)
                col = gn.linmap[Name(lins[j])]
                c1[j] = gn.dtarget[1,col]
                c2[j] = gn.dtarget[2,col]
            end
            for (i, _) ∈ enumerate(gn.chillins)
                pres[2i-1,k] = gn.present[1,i]
                pres[2i,k] = gn.present[2,i]
            end
        end
    end
    Tree(Float64(timezero(g)), Float64(g.time), time, tend, type, par, ch1, ch2,
         nch, nlin, W, ptr, lins, !isnothing(guide), c1, c2, A, κ, pres)
end

## ----------------------------------------------------------------------
## particle population (struct of arrays)

mutable struct Particles
    S::Vector{Int}
    E::Vector{Int}
    I::Vector{Int}
    R::Vector{Int}
    lE::Vector{Int}
    lI::Vector{Int}
    col::Matrix{UInt64}         # W × Np; bit set ⇔ lineage in deme I
    logw::Vector{Float64}
end

Particles(Np::Int, W::Int) = Particles(
    zeros(Int,Np), zeros(Int,Np), zeros(Int,Np), zeros(Int,Np),
    zeros(Int,Np), zeros(Int,Np), zeros(UInt64,W,Np), zeros(Float64,Np))

@inline getbit(col, i, b) = (col[((b-1)>>6)+1, i] >> ((b-1)&63)) & one(UInt64) == one(UInt64)
@inline setbit!(col, i, b) = (col[((b-1)>>6)+1, i] |= one(UInt64) << ((b-1)&63); nothing)
@inline clrbit!(col, i, b) = (col[((b-1)>>6)+1, i] &= ~(one(UInt64) << ((b-1)&63)); nothing)

copyparticle!(dst::Particles, j, src::Particles, i) = begin
    dst.S[j] = src.S[i]; dst.E[j] = src.E[i]; dst.I[j] = src.I[i]; dst.R[j] = src.R[i]
    dst.lE[j] = src.lE[i]; dst.lI[j] = src.lI[i]
    @inbounds for w ∈ axes(src.col,1)
        dst.col[w,j] = src.col[w,i]
    end
    dst.logw[j] = 0.0
    nothing
end

## ----------------------------------------------------------------------
## options

"""
    Options(; proposal = :local, guided = true, order = :post,
              resample = :every, trigger = 0.5, nchunks = 64, threaded = true)

- `proposal`: base weights of the lineage colour-change proposal at a
  population jump:
  - `:naive` — paper/C naive choice (π∅ ∝ n−ℓ, π_b ∝ 1); violates the support
    condition for transmissions when I = ℓ_I (kept to reproduce that bias),
  - `:fixed` — as `:naive` but π∅ ∝ n−ℓ+1 for transmissions (support-correct),
  - `:local` — π ∝ target boost φ (locally optimal; support-correct).
- `guided`: multiply the swap weight of lineage b by the guide's relative
  hazard r_b(t) evaluated at the jump time.
- `order`: `:pre` applies each genealogical event after resampling (as in the
  POMP-based filters); `:post` applies it before resampling.
- `resample`: `:every` or `:ess` (resample when ESS < `trigger`·Np).
"""
Base.@kwdef struct Options
    proposal::Symbol = :local
    guided::Bool = true
    order::Symbol = :post
    resample::Symbol = :every
    trigger::Float64 = 0.5
    nchunks::Int = 64
    threaded::Bool = true
end

## ----------------------------------------------------------------------
## guide evaluation

## returns (p_E, p_I) for the lineage stored at position j of interval k at time t
@inline guideprobs(tr::Tree, j::Int, e::Float64) = begin
    a11, a21, a12, a22 = tr.A
    pE = a11*tr.c1[j] + a12*e*tr.c2[j]
    pI = a21*tr.c1[j] + a22*e*tr.c2[j]
    max(pE, 0.0), max(pI, 0.0)
end

@inline relhaz(tr::Tree, j::Int, e::Float64, toI::Bool) = begin
    pE, pI = guideprobs(tr, j, e)
    toI ? pI/pE : pE/pI
end

## ----------------------------------------------------------------------
## regular part: Gillespie for the target driver; colour changes proposed

## choose the outcome of a transmission (toI = false) or progression (toI = true)
## among "no change" (returns 0) and lineages in the source deme; returns
## (lineage, log boost)
@inline choose_outcome(tr::Tree, P::Particles, i::Int, k::Int, t::Float64,
                       opt::Options, rng::AbstractRNG, toI::Bool) = begin
    S, E, I = P.S[i], P.E[i], P.I[i]
    lE, lI = P.lE[i], P.lI[i]
    ## target boosts φ (post-jump counts)
    if toI      # progression E→I: post (E-1, I+1)
        φ0 = E-1 ≥ lE ? (I+1-lI)/(I+1) : 0.0
        φb = 1.0/(I+1)
        n, ℓ = E, lE
    else        # transmission: post (E+1, I)
        φ0 = (E+1-lE)/(E+1)
        φb = (I-lI+1)/((E+1)*I)
        n, ℓ = I, lI
    end
    ## base proposal weights
    if opt.proposal === :naive
        q0, qb = Float64(n-ℓ), 1.0
    elseif opt.proposal === :fixed
        q0, qb = Float64(n-ℓ+(toI ? 0 : 1)), 1.0
    else  # :local
        q0, qb = φ0, φb
    end
    e = tr.guided && opt.guided ? exp(-tr.κ*(tr.tend[k]-t)) : 1.0
    ## total swap weight
    tot = q0
    @inbounds for j ∈ tr.ptr[k]:(tr.ptr[k+1]-1)
        b = tr.lins[j]
        if getbit(P.col, i, b) != toI
            r = tr.guided && opt.guided ? relhaz(tr, j, e, toI) : 1.0
            tot += qb*r
        end
    end
    tot > 0 || return (Int32(-1), -Inf)
    u = rand(rng)*tot
    if u < q0
        return (Int32(0), log(φ0*tot/q0))
    end
    u -= q0
    last = Int32(-1); lastw = 0.0
    @inbounds for j ∈ tr.ptr[k]:(tr.ptr[k+1]-1)
        b = tr.lins[j]
        if getbit(P.col, i, b) != toI
            r = tr.guided && opt.guided ? relhaz(tr, j, e, toI) : 1.0
            w = qb*r
            last = b; lastw = w
            if u < w
                return (b, log(φb*tot/w))
            end
            u -= w
        end
    end
    (last, log(φb*tot/lastw))    # floating-point fallthrough
end

regular!(tr::Tree, P::Particles, i::Int, k::Int, t::Float64, tf::Float64,
         p, opt::Options, rng::AbstractRNG) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    logw = P.logw[i]
    nev = 0
    while true
        S, E, I, R = P.S[i], P.E[i], P.I[i], P.R[i]
        lI = P.lI[i]
        a1 = β*S*I/pop
        a2 = σ*E
        a3 = I > lI ? γ*I : 0.0
        a4 = ω*R
        dec = (ψ+χ)*I + (I > lI ? 0.0 : γ*I)
        A = a1+a2+a3+a4
        step = A > 0 ? randexp(rng)/A : Inf
        if t + step ≥ tf
            logw -= dec*(tf-t)
            break
        end
        logw -= dec*step
        t += step
        nev += 1
        u = rand(rng)*A
        if u < a1
            b, lb = choose_outcome(tr, P, i, k, t, opt, rng, false)
            if b < 0
                logw = -Inf; break
            end
            logw += lb
            P.S[i] -= 1; P.E[i] += 1
            if b > 0
                clrbit!(P.col, i, b); P.lI[i] -= 1; P.lE[i] += 1
            end
        elseif u < a1+a2
            b, lb = choose_outcome(tr, P, i, k, t, opt, rng, true)
            if b < 0
                logw = -Inf; break
            end
            logw += lb
            P.E[i] -= 1; P.I[i] += 1
            if b > 0
                setbit!(P.col, i, b); P.lE[i] -= 1; P.lI[i] += 1
            end
        elseif u < a1+a2+a3
            P.I[i] -= 1; P.R[i] += 1
        else
            P.R[i] -= 1; P.S[i] += 1
        end
        isfinite(logw) || break
    end
    P.logw[i] = logw
    nev
end

## ----------------------------------------------------------------------
## singular part: genealogical event k

singular!(tr::Tree, P::Particles, i::Int, k::Int, p, opt::Options, rng::AbstractRNG) = begin
    (; β, ψ, χ, pop) = p
    S, E, I = P.S[i], P.E[i], P.I[i]
    lE, lI = P.lE[i], P.lI[i]
    typ = tr.type[k]
    guided = tr.guided && opt.guided
    if typ == ROOT
        c = tr.ch1[k]
        gE = guided ? tr.pres[1,k] : 1.0
        gI = guided ? tr.pres[2,k] : 1.0
        wE = E-lE ≥ 1 ? gE : 0.0
        wI = I-lI ≥ 1 ? gI : 0.0
        tot = wE+wI
        if tot ≤ 0
            P.logw[i] = -Inf; return nothing
        end
        if rand(rng)*tot < wE
            P.lE[i] += 1; P.logw[i] += log(tot/wE)
        else
            setbit!(P.col, i, c); P.lI[i] += 1; P.logw[i] += log(tot/wI)
        end
        return nothing
    end
    b = tr.par[k]
    if !getbit(P.col, i, b)
        P.logw[i] = -Inf; return nothing
    end
    clrbit!(P.col, i, b); P.lI[i] -= 1
    lI2 = lI-1
    if typ == SAMPLE && tr.nch[k] == 0
        w1 = ψ*(I-lI2)
        w2 = χ*I
        tot = w1+w2
        if tot ≤ 0
            P.logw[i] = -Inf; return nothing
        end
        P.logw[i] += log(tot)
        if rand(rng)*tot ≥ w1
            P.I[i] -= 1
        end
    elseif typ == SAMPLE
        setbit!(P.col, i, tr.ch1[k]); P.lI[i] += 1
        P.logw[i] += log(ψ)
    else  # branch: parent in I gives one child in I and one in E
        if S < 1
            P.logw[i] = -Inf; return nothing
        end
        wt = β*S/(pop*(E+1))
        g1 = guided ? tr.pres[2,k]*tr.pres[3,k] : 1.0  # c1 in I, c2 in E
        g2 = guided ? tr.pres[1,k]*tr.pres[4,k] : 1.0  # c1 in E, c2 in I
        tot = g1+g2
        c1, c2 = tr.ch1[k], tr.ch2[k]
        if rand(rng)*tot < g1
            setbit!(P.col, i, c1); P.logw[i] += log(wt*tot/g1)
        else
            setbit!(P.col, i, c2); P.logw[i] += log(wt*tot/g2)
        end
        P.lI[i] += 1; P.lE[i] += 1
        P.S[i] -= 1; P.E[i] += 1
    end
    nothing
end

## ----------------------------------------------------------------------
## the filter

"""
    pfilter(tr, p; Np, opt = Options(), rng = default_rng(), diagnostics = false)

Run the particle filter on compiled tree `tr` with parameters `p` (a named
tuple with `β, σ, γ, ω, ψ, χ, pop, S0, E0, I0, R0`).  Returns a named tuple
with the log-likelihood estimate `logLik` and, if `diagnostics = true`,
per-event vectors `ess`, `zerofrac`, `cll` and the `type` of each event.
"""
pfilter(tr::Tree, p::NamedTuple; Np::Int, opt::Options = Options(),
        rng::AbstractRNG = default_rng(), diagnostics::Bool = false) = begin
    M = length(tr.time)
    x0 = POMP.barycentric(Int64, (p.S0, p.E0, p.I0, p.R0), p.pop)
    P = Particles(Np, tr.W); Q = Particles(Np, tr.W)
    P.S .= x0[1]; P.E .= x0[2]; P.I .= x0[3]; P.R .= x0[4]
    ess = diagnostics ? fill(NaN, M) : Float64[]
    zf = diagnostics ? fill(NaN, M) : Float64[]
    cll = diagnostics ? fill(NaN, M) : Float64[]
    nev = zeros(Int, opt.nchunks)
    W = fill(1.0/Np, Np)          # normalized weights carried between steps
    ancestors = Vector{Int}(undef, Np)
    loglik = 0.0
    seed = rand(rng, UInt64)
    post = opt.order === :post
    ## with :post ordering, leading events at t0 (roots) are applied at the start
    k0 = 1
    if post
        while k0 ≤ M && tr.time[k0] == tr.t0 && tr.type[k0] == ROOT
            run_chunks!(P, Np, opt, seed, k0, nev) do i, r
                singular!(tr, P, i, k0, p, opt, r)
            end
            k0 += 1
        end
        k0 -= 1     # interval k0 is the last root's interval
    end
    for k ∈ max(k0,1):M
        ## propagate: (pre: event k) + regular part over interval k + (post: event k+1)
        run_chunks!(P, Np, opt, seed, k, nev) do i, r
            if !post
                singular!(tr, P, i, k, p, opt, r)
            end
            if isfinite(P.logw[i])
                regular!(tr, P, i, k, tr.time[k], tr.tend[k], p, opt, r)
            end
            if post && k < M && isfinite(P.logw[i])
                singular!(tr, P, i, k+1, p, opt, r)
            end
        end
        ## weights
        mx = maximum(P.logw)
        if !isfinite(mx)
            return (logLik = -Inf, ess, zerofrac = zf, cll, type = tr.type, nevents = sum(nev))
        end
        s = 0.0; ss = 0.0; nz = 0
        @inbounds for i ∈ 1:Np
            v = W[i]*exp(P.logw[i]-mx)
            W[i] = v
            s += v; ss += v*v
            nz += P.logw[i] == -Inf
        end
        ll = mx + log(s)
        loglik += ll
        W ./= s
        e = 1.0/sum(abs2, W)
        if diagnostics
            ke = post ? min(k+1, M) : k
            ess[ke] = e; zf[ke] = nz/Np; cll[ke] = ll
        end
        P.logw .= 0.0
        if opt.resample === :every || e < opt.trigger*Np
            systematic!(ancestors, W, rng)
            @inbounds for j ∈ 1:Np
                copyparticle!(Q, j, P, ancestors[j])
            end
            P, Q = Q, P
            W .= 1.0/Np
        end
    end
    (logLik = loglik, ess, zerofrac = zf, cll, type = tr.type, nevents = sum(nev))
end

## run f(i, rng) for all particles, in parallel chunks with reproducible RNGs
run_chunks!(f, P::Particles, Np::Int, opt::Options, seed::UInt64, k::Int, nev) = begin
    nc = min(opt.nchunks, Np)
    if opt.threaded && nthreads() > 1
        @threads :static for c ∈ 1:nc
            rng = Xoshiro(seed, UInt64(k), UInt64(c))
            for i ∈ c:nc:Np
                f(i, rng)
            end
        end
    else
        for c ∈ 1:nc
            rng = Xoshiro(seed, UInt64(k), UInt64(c))
            for i ∈ c:nc:Np
                f(i, rng)
            end
        end
    end
    nothing
end

systematic!(a::Vector{Int}, w::Vector{Float64}, rng::AbstractRNG) = begin
    n = length(w)
    u = rand(rng)/n
    c = w[1]; j = 1
    @inbounds for i ∈ 1:n
        while u > c && j < n
            j += 1; c += w[j]
        end
        a[i] = j
        u += 1.0/n
    end
    a
end

"""
    logliks(tr, p; Np, n, opt, rng)

`n` independent log-likelihood estimates.
"""
logliks(tr::Tree, p::NamedTuple; Np::Int, n::Int, opt::Options = Options(),
        rng::AbstractRNG = default_rng()) =
    [pfilter(tr, p; Np, opt, rng).logLik for _ ∈ 1:n]

end
