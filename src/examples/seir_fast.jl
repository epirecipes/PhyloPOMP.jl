"""
    FastSEIR

A streamlined implementation of the phylodynamic filter for an SEIR
model.  It estimates the same likelihood as the other SEIR filters, but

- the genealogy is compiled once into arrays shared by all particles
  (see [`Tree`](@ref PhyloPOMP.FastSEIR.Tree)),
- each particle's coloring is a bitmask, and only the current particles
  are kept, so memory grows only with the number of particles,
- a lineage color change accompanying a population event is proposed
  with probability proportional to its exact weight, optionally
  multiplied by the relative hazard given by a filter guide,
- each genealogical event is applied before, rather than after,
  resampling, so that particles incompatible with it are not carried
  forward, and
- particles are propagated in parallel, with results that do not depend
  on the number of threads.

Run the filter with `pfilter` on a `Tree`; estimate parameters by
iterated filtering with `mif`.
"""
module FastSEIR

using Random: AbstractRNG, Xoshiro, default_rng, randexp
using Base.Threads: @threads, nthreads
using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time, NodeType, FSMarkovProc
import PartiallyObservedMarkovProcesses as POMP

## Construct an enumeration of the demes for this model.
@demes Demes Expos Infec
using .Demes: Expos, Infec

include("seir_trees.jl")

"""
    knowledge!(v; deme, type, time)

Lineages are infectious at samples and branch points.
"""
knowledge!(
    v; deme, type, time,
) = begin
    if type==Sample || type==Node
        demekron!(v,Infec)
        true
    else
        false
    end
end

"""
    Tree(g)
    Tree(g, m)

Compile the [`Genealogy`](@ref) `g` for use by `pfilter`.  If a
finite-state Markov process `m` on `FastSEIR.Demes` is given (see
[`fsmarkov`](@ref)), a filter guide is constructed (see [`guide`](@ref))
and used to steer the proposals.  Interval `k` runs from the time of
event `k` to that of event `k+1` (or to the end of the genealogy).
"""
struct Tree
    "times of the genealogical events"
    time::Vector{Time}
    "right ends of the intervals"
    tend::Vector{Time}
    "types of the events"
    type::Vector{NodeType}
    "parent lineage of each event"
    parlin::Vector{Name}
    "child lineages of each event"
    chillins::Vector{Vector{Name}}
    "the lineages present on interval `k` are `lins[ptr[k]:ptr[k+1]-1]`"
    ptr::Vector{Int}
    lins::Vector{Name}
    "number of 64-bit words in a coloring"
    nwords::Int
    "whether a guide is available"
    guided::Bool
    "guide probabilities of the demes of the child lineages of each event"
    present::Vector{Matrix{Float64}}
    "guide coefficients of the lineages in `lins`"
    c1::Vector{Float64}
    c2::Vector{Float64}
    "guide process: probabilities `A[:,1]*c1 + A[:,2]*exp(-κ*(tend-t))*c2`"
    A::Matrix{Float64}
    κ::Float64
end

Tree(g::Genealogy, m::FSMarkovProc) = Tree(g, guide(g,m,knowledge!))

Tree(
    g::Genealogy,
    gd::Union{Guide,Nothing} = nothing,
) = begin
    M = length(g)
    @assert M > 0 "empty genealogy"
    time = Time[g[k].slate for k ∈ 1:M]
    tend = Time[time[2:end]..., g.time]
    type = NodeType[g[k].type for k ∈ 1:M]
    parlin = Name[g[k].lineage for k ∈ 1:M]
    chillins = [Name[g[c].lineage for c ∈ g[k].children] for k ∈ 1:M]
    ptr = Vector{Int}(undef,M+1)
    lins = Name[]
    present = Set{Name}()
    for k ∈ 1:M
        delete!(present,parlin[k])
        union!(present,chillins[k])
        ptr[k] = length(lins)+1
        append!(lins,sort!(collect(present)))
    end
    ptr[M+1] = length(lins)+1
    nwords = cld(max(Int(g.nsample),1),64)
    if isnothing(gd)
        pres = [ones(2,length(chillins[k])) for k ∈ 1:M]
        c1 = zeros(length(lins))
        c2 = zeros(length(lins))
        Tree(time,tend,type,parlin,chillins,ptr,lins,nwords,false,pres,c1,c2,zeros(2,2),0.0)
    else
        m = gd.process
        @assert length(m.eigenvals) == 2 "the guide must have two demes"
        col(k,j) = gd[k].linmap[lins[j]]
        c1 = [gd[k].dtarget[1,col(k,j)] for k ∈ 1:M for j ∈ ptr[k]:ptr[k+1]-1]
        c2 = [gd[k].dtarget[2,col(k,j)] for k ∈ 1:M for j ∈ ptr[k]:ptr[k+1]-1]
        pres = [gd[k].present for k ∈ 1:M]
        Tree(time,tend,type,parlin,chillins,ptr,lins,nwords,true,pres,c1,c2,m.A,-m.eigenvals[2])
    end
end

## guide probabilities (E, I) of the lineage at position `j` of `tr.lins`,
## where `e = exp(-κ*(tend-t))`
guideprob(tr::Tree, j::Integer, e::Real) = begin
    pE = tr.A[1,1]*tr.c1[j] + tr.A[1,2]*e*tr.c2[j]
    pI = tr.A[2,1]*tr.c1[j] + tr.A[2,2]*e*tr.c2[j]
    max(pE,0.0), max(pI,0.0)
end

## The particles, stored as a structure of arrays.  In the coloring of
## particle `i`, bit `b-1` of column `i` of `col` is set if and only if
## lineage `b` is in deme I.
struct Particles
    S::Vector{Int}
    E::Vector{Int}
    I::Vector{Int}
    R::Vector{Int}
    ellE::Vector{Int}
    ellI::Vector{Int}
    col::Matrix{UInt64}
    ll::Vector{Float64}
end

Particles(Np::Integer, nwords::Integer) = Particles(
    zeros(Int,Np), zeros(Int,Np), zeros(Int,Np), zeros(Int,Np),
    zeros(Int,Np), zeros(Int,Np), zeros(UInt64,nwords,Np), zeros(Np),
)

word(b) = (Int(b)-1)>>6 + 1
bit(b) = one(UInt64) << ((Int(b)-1)&63)
isinfec(P::Particles, i, b) = P.col[word(b),i] & bit(b) ≠ 0
infec!(P::Particles, i, b) = (P.col[word(b),i] |= bit(b); nothing)
expos!(P::Particles, i, b) = (P.col[word(b),i] &= ~bit(b); nothing)

copyparticle!(Q::Particles, j, P::Particles, i) = begin
    Q.S[j] = P.S[i]; Q.E[j] = P.E[i]; Q.I[j] = P.I[i]; Q.R[j] = P.R[i]
    Q.ellE[j] = P.ellE[i]; Q.ellI[j] = P.ellI[i]
    Q.col[:,j] .= @view(P.col[:,i])
    Q.ll[j] = 0.0
    nothing
end

## proposal weight of a change of deme of the lineage at position `j` of
## `tr.lins` (zero if it is already in the destination deme)
swap_weight(tr::Tree, P::Particles, i, j, toI::Bool, qb, e, guided::Bool) = begin
    if isinfec(P,i,tr.lins[j]) == toI
        0.0
    elseif guided
        pE, pI = guideprob(tr,j,e)
        qb*(toI ? pI/pE : pE/pI)
    else
        qb
    end
end

## Propose the change of coloring that accompanies a transmission (a
## lineage may pass from I to E; `toI = false`) or a progression (a
## lineage may pass from E to I; `toI = true`) in particle `i`, with
## counts `E`, `I`, `ellE`, `ellI`, at time `t` in interval `k`.  Returns
## the lineage that changes deme (zero for none) and the log importance
## weight.
color_change(
    tr::Tree, P::Particles, i, k, t, toI::Bool, E, I, ellE, ellI, rng;
    proposal, guided,
) = begin
    ## exact weights of no change and of a change of any one lineage
    φ0, φb = toI ?
        (@indicator(E > ellE, (I+1-ellI)/(I+1)), 1/(I+1)) :
        ((E+1-ellE)/(E+1), (I+1-ellI)/((E+1)*I))
    ## proposal weights
    q0, qb = proposal == :local ? (φ0, φb) : (Float64(toI ? E-ellE : I-ellI+1), 1.0)
    e = guided ? exp(-tr.κ*(tr.tend[k]-t)) : 1.0
    lins = tr.ptr[k]:(tr.ptr[k+1]-1)
    total = q0
    for j ∈ lins
        total += swap_weight(tr,P,i,j,toI,qb,e,guided)
    end
    u = rand(rng)*total
    if u < q0
        return zero(Name), log(φ0*total/q0)
    end
    u -= q0
    jb, wb = 0, 0.0
    for j ∈ lins
        w = swap_weight(tr,P,i,j,toI,qb,e,guided)
        if w > 0
            jb, wb = j, w
            u < w && break
            u -= w
        end
    end
    tr.lins[jb], log(φb*total/wb)
end

## Simulate the population process of particle `i` over interval `k`
## (Gillespie algorithm), proposing color changes along the way.  The
## state is kept in local variables and stored at the end.  Returns the
## number of population events.
regular_part!(
    tr::Tree, P::Particles, i, k, p, rng;
    proposal, guided,
) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    S, E, I, R = P.S[i], P.E[i], P.I[i], P.R[i]
    ellE, ellI = P.ellE[i], P.ellI[i]
    ll = P.ll[i]
    nevent = 0
    while true
        transmission = β*S*I/pop
        progression = σ*E
        recovery = @indicator(I > ellI, γ*I)
        waning = ω*R
        decay = (ψ+χ)*I + @indicator(I ≤ ellI, γ*I)
        rate = transmission+progression+recovery+waning
        step = randexp(rng)/rate
        if t+step ≥ tf
            ll -= decay*(tf-t)
            break
        end
        ll -= decay*step
        t += step
        nevent += 1
        u = rand(rng)*rate
        if u < transmission
            b, lw = color_change(tr,P,i,k,t,false,E,I,ellE,ellI,rng;proposal,guided)
            ll += lw
            S -= 1; E += 1
            if b > 0
                expos!(P,i,b); ellI -= 1; ellE += 1
            end
        elseif u < transmission+progression
            b, lw = color_change(tr,P,i,k,t,true,E,I,ellE,ellI,rng;proposal,guided)
            ll += lw
            E -= 1; I += 1
            if b > 0
                infec!(P,i,b); ellE -= 1; ellI += 1
            end
        elseif u < transmission+progression+recovery
            I -= 1; R += 1
        else
            R -= 1; S += 1
        end
    end
    P.S[i], P.E[i], P.I[i], P.R[i] = S, E, I, R
    P.ellE[i], P.ellI[i] = ellE, ellI
    P.ll[i] = ll
    nevent
end

## Apply genealogical event `k` to particle `i`.
singular_part!(
    tr::Tree, P::Particles, i, k, p, rng;
    guided,
) = begin
    (; β, ψ, χ, pop) = p
    S, E, I, ellE, ellI = P.S[i], P.E[i], P.I[i], P.ellE[i], P.ellI[i]
    g(d,c) = guided ? tr.present[k][d,c] : 1.0
    chillins = tr.chillins[k]
    if tr.type[k] == Root               # the new lineage goes into a deme with room
        wE = @indicator(E > ellE, g(1,1))
        wI = @indicator(I > ellI, g(2,1))
        total = wE+wI
        if total == 0
            P.ll[i] = -Inf
        elseif rand(rng)*total < wE
            P.ellE[i] += 1
            P.ll[i] += log(total/wE)
        else
            infec!(P,i,chillins[1]); P.ellI[i] += 1
            P.ll[i] += log(total/wI)
        end
    elseif !isinfec(P,i,tr.parlin[k])   # the parent must be infectious
        P.ll[i] = -Inf
    elseif tr.type[k] == Sample && isempty(chillins)
        expos!(P,i,tr.parlin[k]); P.ellI[i] -= 1
        nondestructive = ψ*(I-ellI+1)
        total = nondestructive + χ*I
        P.ll[i] += log(total)
        if rand(rng)*total ≥ nondestructive
            P.I[i] -= 1
        end
    elseif tr.type[k] == Sample         # the sampled lineage continues
        expos!(P,i,tr.parlin[k])
        infec!(P,i,chillins[1])
        P.ll[i] += log(ψ)
    elseif S == 0
        P.ll[i] = -Inf
    else                                # branch: one child in I, the other in E
        expos!(P,i,tr.parlin[k])
        w1 = g(2,1)*g(1,2)              # first child in I, second in E
        w2 = g(1,1)*g(2,2)              # first child in E, second in I
        total = w1+w2
        if rand(rng)*total < w1
            infec!(P,i,chillins[1]); P.ll[i] += log(total/w1)
        else
            infec!(P,i,chillins[2]); P.ll[i] += log(total/w2)
        end
        P.ll[i] += log(β*S/(pop*(E+1)))
        P.ellE[i] += 1
        P.S[i] -= 1; P.E[i] += 1
    end
    nothing
end

"""
    FastPfilterd

The result of running `pfilter` on a [`Tree`](@ref PhyloPOMP.FastSEIR.Tree).
`logLik`, `eff_sample_size`, and `cond_logLik` extract the log likelihood
estimate and, for each interval, the effective sample size and the
conditional log likelihood.
"""
struct FastPfilterd
    "log likelihood estimate"
    logLik::Float64
    "effective sample size at the end of each interval"
    eff_sample_size::Vector{Float64}
    "conditional log likelihood of each interval"
    cond_logLik::Vector{Float64}
    "fraction of particles with zero weight at the end of each interval"
    zero_fraction::Vector{Float64}
    "number of population events simulated"
    nevents::Int
end

POMP.logLik(pf::FastPfilterd) = pf.logLik
POMP.eff_sample_size(pf::FastPfilterd) = pf.eff_sample_size
POMP.cond_logLik(pf::FastPfilterd) = pf.cond_logLik

"""
    pfilter(tr::Tree; Np = 1,
            β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
            pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
            proposal = :local, guided = true, trigger = 1,
            threaded = true, rng = default_rng())

Run the particle filter with `Np` particles on the compiled genealogy
`tr`.  The model parameters have the same meanings and defaults as in
the `filter_pomp` constructors of the other SEIR modules.

- `proposal`: with `:local`, a change of coloring is proposed with
  probability proportional to its exact weight; with `:uniform`, in
  proportion to the number of individuals involved.
- `guided`: if `tr` has a guide, multiply the proposal weight of each
  lineage's change of deme by the guide's relative hazard.
- `trigger`: resample whenever the effective sample size falls below
  `trigger*Np` (by default, after every genealogical event).
- `threaded`: propagate the particles on all available threads.  The
  results depend on `rng` but not on the number of threads.

Returns a [`FastPfilterd`](@ref PhyloPOMP.FastSEIR.FastPfilterd) object.
"""
POMP.pfilter(
    tr::Tree;
    Np::Integer = 1,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    proposal::Symbol = :local,
    guided::Bool = true,
    trigger::Real = 1,
    threaded::Bool = true,
    rng::AbstractRNG = default_rng(),
) = begin
    @assert proposal ∈ (:local, :uniform) "unknown proposal '$proposal'"
    useguide = guided && tr.guided
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords)
    Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)                   # normalized weights
    ess = fill(Float64(Np),M)
    cll = zeros(M)
    zerofrac = zeros(M)
    ancestors = Vector{Int}(undef,Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    ## apply the events at the initial time
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        foreach_particle(P,seed,-j,threaded) do P, i, rng
            singular_part!(tr,P,i,j,p,rng;guided=useguide)
            0
        end
    end
    ## step k: the regular part on interval k, then event k+1
    for k ∈ k0:M
        nevents += foreach_particle(P,seed,k,threaded) do P, i, rng
            (W[i] > 0 && isfinite(P.ll[i])) || return 0
            n = regular_part!(tr,P,i,k,p,rng;proposal,guided=useguide)
            k < M && singular_part!(tr,P,i,k+1,p,rng;guided=useguide)
            n
        end
        llmax = maximum(P.ll)
        isfinite(llmax) || return FastPfilterd(-Inf,ess,cll,zerofrac,nevents)
        W .*= exp.(P.ll .- llmax)
        s = sum(W)
        cll[k] = llmax+log(s)
        loglik += cll[k]
        W ./= s
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/Np
        P.ll .= 0.0
        if ess[k] ≤ trigger*Np
            systematic!(ancestors,W,rng)
            copyparticles!(Q,P,ancestors,threaded)
            P, Q = Q, P
            W .= 1/Np
        end
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

"""
    FastMifd

The result of running `mif` on a [`Tree`](@ref PhyloPOMP.FastSEIR.Tree).
`coef` returns the parameter estimate, `traces` the parameters and log
likelihood at each iteration, and `logLik`, `eff_sample_size`, and
`cond_logLik` the results of a final `pfilter` at the estimate.
"""
struct FastMifd{P<:NamedTuple}
    "result of `pfilter` at the final parameter estimate"
    pf::FastPfilterd
    "averages of the parameters at the start and after each iteration"
    trace::Vector{P}
    "log likelihood of the perturbed filter at each iteration"
    logLik::Vector{Float64}
end

POMP.logLik(m::FastMifd) = logLik(m.pf)
POMP.eff_sample_size(m::FastMifd) = eff_sample_size(m.pf)
POMP.cond_logLik(m::FastMifd) = cond_logLik(m.pf)
POMP.coef(m::FastMifd) = m.trace[end]
POMP.traces(m::FastMifd) = POMP.melt(
    [merge(p,(logLik=l,)) for (p,l) ∈ zip(m.trace,[m.logLik...,missing])],
    :iteration,
)

"""
    mif(tr::Tree; Np = 1, Nmif = 1, perturbations, cooling, avfun = geomean,
        β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
        pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
        proposal = :local, guided = true, threaded = true,
        rng = default_rng())

Iterated filtering (IF2) with the filter of `pfilter`, starting from the
given model parameters.  As for the `mif` method on `PompObject`s, each
particle carries its own parameters, which are replaced by
`perturbations(scale, lag; params...)` at lag 0 and after each
genealogical event, with `scale = cooling(n)` in iteration `n+1`, and
the parameters are resampled along with the particles.  `perturbations`
may be constructed with `@perturbn`; `cooling` with `geometric_cooling`.
The population size `pop` is rounded to an integer.  The perturbations
draw on the global random-number generator.

Returns a [`FastMifd`](@ref PhyloPOMP.FastSEIR.FastMifd) object.
"""
POMP.mif(
    tr::Tree;
    Np::Integer = 1,
    Nmif::Integer = 1,
    perturbations::Function,
    cooling::Function,
    avfun::Function = POMP.geomean,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    proposal::Symbol = :local,
    guided::Bool = true,
    threaded::Bool = true,
    rng::AbstractRNG = default_rng(),
) = begin
    @assert proposal ∈ (:local, :uniform) "unknown proposal '$proposal'"
    useguide = guided && tr.guided
    θ = map(Float64, (;β,σ,γ,ω,ψ,χ,pop,S0,E0,I0,R0))
    M = length(tr.time)
    params = fill(θ,Np)
    work = similar(params)
    P = Particles(Np,tr.nwords)
    Q = Particles(Np,tr.nwords)
    W = zeros(Np)
    ancestors = Vector{Int}(undef,Np)
    trace = [θ]
    logliks = Float64[]
    k0 = findlast(==(tr.time[1]),tr.time)
    for n ∈ 1:Nmif
        scale = cooling(n-1)
        perturb!(params,perturbations,scale,0)
        initialize!(P,params)
        seed = rand(rng,UInt64)
        for j ∈ 1:k0
            foreach_particle(P,seed,-j,threaded) do P, i, rng
                singular_part!(tr,P,i,j,rates(params[i]),rng;guided=useguide)
                0
            end
        end
        loglik = 0.0
        for k ∈ k0:M
            foreach_particle(P,seed,k,threaded) do P, i, rng
                isfinite(P.ll[i]) || return 0
                p = rates(params[i])
                regular_part!(tr,P,i,k,p,rng;proposal,guided=useguide)
                k < M && singular_part!(tr,P,i,k+1,p,rng;guided=useguide)
                0
            end
            llmax = maximum(P.ll)
            isfinite(llmax) || error("in iteration $n, no particle is compatible with event $(k+1)")
            W .= exp.(P.ll .- llmax)
            s = sum(W)
            loglik += llmax+log(s/Np)
            W ./= s
            systematic!(ancestors,W,rng)
            copyparticles!(Q,P,ancestors,threaded)
            for (j,a) ∈ enumerate(ancestors)
                work[j] = params[a]
            end
            P, Q = Q, P
            params .= work
            if k < M || n < Nmif
                perturb!(params,perturbations,scale,k-k0+1)
            end
        end
        push!(logliks,loglik)
        push!(trace,average(params,avfun))
    end
    est = trace[end]
    pf = pfilter(tr; Np, est..., pop=round(Int,est.pop), proposal, guided, threaded, rng)
    FastMifd(pf,trace,logliks)
end

## model rates of a particle's parameters
rates(θ) = (β=θ.β, σ=θ.σ, γ=θ.γ, ω=θ.ω, ψ=θ.ψ, χ=θ.χ, pop=round(θ.pop))

## set each particle to its initial state, with no lineages
initialize!(P::Particles, params) = begin
    P.ellE .= 0; P.ellI .= 0; P.col .= 0; P.ll .= 0.0
    for (i,θ) ∈ enumerate(params)
        P.S[i], P.E[i], P.I[i], P.R[i] =
            barycentric(Int64,(θ.S0,θ.E0,θ.I0,θ.R0),round(Int,θ.pop))
    end
    P
end

## apply the perturbation kernel to each particle's parameters
perturb!(params, f, scale, lag) = begin
    for i ∈ eachindex(params)
        params[i] = merge(params[i],map(Float64,f(scale,lag;params[i]...)))
    end
    params
end

## average of the parameters over the particles
average(params, avfun) = begin
    names = keys(first(params))
    NamedTuple{names}(map(n -> avfun(getfield.(params,n)),names))
end

## Apply `f(P, i, rng)` to each particle `i` and return the sum of the
## results.  The particles are dealt into chunks, each with its own
## random-number stream determined by `seed` and `stream`, so that the
## results do not depend on the number of threads.  Dealing (rather than
## dividing) spreads the copies of each resampled particle, which are
## adjacent, over the chunks, and so balances the work.
foreach_particle(
    f, P::Particles, seed, stream, threaded::Bool,
) = begin
    Np = length(P.ll)
    nchunk = min(64,Np)
    counts = zeros(Int,nchunk)
    foreach_chunk(nchunk,threaded) do c
        rng = Xoshiro(hash((seed,stream,c)))
        n = 0
        for i ∈ c:nchunk:Np
            n += f(P,i,rng)
        end
        counts[c] = n
    end
    sum(counts)
end

## copy particle `ancestors[j]` of `P` into position `j` of `Q`, for each `j`
copyparticles!(
    Q::Particles, P::Particles, ancestors, threaded::Bool,
) = begin
    Np = length(ancestors)
    nchunk = min(64,Np)
    foreach_chunk(nchunk,threaded) do c
        for j ∈ chunk(c,nchunk,Np)
            copyparticle!(Q,j,P,ancestors[j])
        end
    end
    Q
end

## the particles in chunk `c` of `nchunk`
chunk(c, nchunk, Np) = div((c-1)*Np,nchunk)+1:div(c*Np,nchunk)

## Call `work(c)` for each chunk `c`, on all available threads if `threaded`.
foreach_chunk(
    work, nchunk, threaded::Bool,
) = begin
    if threaded && nthreads() > 1
        @threads :static for c ∈ 1:nchunk
            work(c)
        end
    else
        foreach(work,1:nchunk)
    end
    nothing
end

## systematic resampling: ancestor indices for normalized weights `w`
systematic!(
    a::Vector{Int}, w::Vector{Float64}, rng::AbstractRNG,
) = begin
    n = length(w)
    u = rand(rng)/n
    c = w[1]
    j = 1
    for i ∈ eachindex(a)
        while u > c && j < n
            j += 1
            c += w[j]
        end
        a[i] = j
        u += 1/n
    end
    a
end

end
