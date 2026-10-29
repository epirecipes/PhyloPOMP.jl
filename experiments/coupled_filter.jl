## Experimental coupled particle filter for FastSEIR, for common random
## numbers across parameter values.
##
## - Within each interval, the population process of the particle in slot j
##   is simulated by the modified next-reaction method: each reaction channel
##   (transmission, progression, recovery, waning) is driven by its own
##   unit-rate Poisson process, whose arrivals, and the uniforms used for the
##   accompanying color changes, come from the channel's own counter-based
##   stream, keyed by (seed, step, slot, channel).  With the same seed, a small
##   change of the parameters then moves event times slightly instead of
##   reshuffling which random number drives which event.
## - Before resampling, the particles are sorted by their state, and systematic
##   resampling with a common uniform is applied in that order, so that each
##   slot receives an ancestor of similar state at nearby parameter values.
##
## Both changes leave the estimator unbiased: the next-reaction method
## simulates the same process as the Gillespie algorithm, and systematic
## resampling is unbiased in any order.
module Coupled

using Random
using Random: AbstractRNG, default_rng
using PhyloPOMP
using PhyloPOMP.FastSEIR: Tree, Particles, FastPfilterd, isinfec, infec!, expos!,
    color_change, regular_part!, singular_part!, copyparticle!, systematic!, foreach_particle

## A counter-based stream of random numbers (SplitMix64 outputs).
mutable struct Stream <: AbstractRNG
    key::UInt64
    n::UInt64
end

splitmix(z::UInt64) = begin
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

rekey!(s::Stream, key) = (s.key = UInt64(key); s.n = 0; s)
Random.rand(s::Stream, ::Random.SamplerTrivial{Random.CloseOpen01_64}) = begin
    s.n += 1
    (splitmix(s.key + s.n*0x9e3779b97f4a7c15) >>> 11) * 0x1.0p-53
end
expo(s::Stream) = -log1p(-rand(s))

const NCHANNEL = 4      # transmission, progression, recovery, waning
const EVENTS = 5        # stream for the genealogical event

## Simulate the population process of particle `i` over interval `k` by the
## modified next-reaction method, with the streams `st[1:4]` of its slot.
coupled_part!(tr::Tree, P::Particles, i, k, p, st; proposal, guided) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    ll = P.ll[i]
    ## internal times of the channels and their next unit-rate arrivals
    T1 = T2 = T3 = T4 = 0.0
    A1, A2, A3, A4 = expo(st[1]), expo(st[2]), expo(st[3]), expo(st[4])
    nevent = 0
    while true
        S, E, I, R, ellI = P.S[i], P.E[i], P.I[i], P.R[i], P.ellI[i]
        a1 = β*S*I/pop
        a2 = σ*E
        a3 = @indicator(I > ellI, γ*I)
        a4 = ω*R
        decay = (ψ+χ)*I + @indicator(I ≤ ellI, γ*I)
        d1 = a1 > 0 ? (A1-T1)/a1 : Inf
        d2 = a2 > 0 ? (A2-T2)/a2 : Inf
        d3 = a3 > 0 ? (A3-T3)/a3 : Inf
        d4 = a4 > 0 ? (A4-T4)/a4 : Inf
        Δ = min(d1,d2,d3,d4)
        if t+Δ ≥ tf
            ll -= decay*(tf-t)
            break
        end
        ll -= decay*Δ
        t += Δ
        T1 += a1*Δ; T2 += a2*Δ; T3 += a3*Δ; T4 += a4*Δ
        nevent += 1
        if Δ == d1
            A1 += expo(st[1])
            b, lw = color_change(tr,P,i,k,t,false,st[1];proposal,guided)
            ll += lw
            P.S[i] -= 1; P.E[i] += 1
            if b > 0
                expos!(P,i,b); P.ellI[i] -= 1; P.ellE[i] += 1
            end
        elseif Δ == d2
            A2 += expo(st[2])
            b, lw = color_change(tr,P,i,k,t,true,st[2];proposal,guided)
            ll += lw
            P.E[i] -= 1; P.I[i] += 1
            if b > 0
                infec!(P,i,b); P.ellE[i] -= 1; P.ellI[i] += 1
            end
        elseif Δ == d3
            A3 += expo(st[3])
            P.I[i] -= 1; P.R[i] += 1
        else
            A4 += expo(st[4])
            P.R[i] -= 1; P.S[i] += 1
        end
    end
    P.ll[i] = ll
    nevent
end

## sort key: the state of particle `i`
statekey(P::Particles, i) = (P.I[i], P.E[i], P.S[i], P.R[i], P.ellI[i], view(P.col,:,i))

## apply the initial event `j` to every particle
initial!(tr::Tree, P::Particles, j, p, streams, seed; guided) = begin
    Threads.@threads :static for i ∈ 1:size(streams,2)
        singular_part!(tr,P,i,j,p,rekey!(streams[EVENTS,i],hash((seed,-j,i)));guided)
    end
end

## step k for every live particle: the regular part on interval k, then event k+1
propagate!(tr::Tree, P::Particles, W, k, p, streams, counts, seed; proposal, guided) = begin
    M = length(tr.time)
    Threads.@threads :static for i ∈ 1:size(streams,2)
        counts[i] = 0
        if W[i] > 0 && isfinite(P.ll[i])
            st = view(streams,:,i)
            for c ∈ 1:EVENTS
                rekey!(st[c],hash((seed,k,i,c)))
            end
            counts[i] = coupled_part!(tr,P,i,k,p,st;proposal,guided)
            k < M && singular_part!(tr,P,i,k+1,p,st[EVENTS];guided)
        end
    end
end

"""
    pfilter_coupled(tr; Np, params..., proposal, guided, sorted, seed)

FastSEIR's filter (resampling after every event) with the coupled
propagation above.  The result is a deterministic function of `seed` and
the parameters.  With `sorted = false` the particles are resampled in slot
order.
"""
pfilter_coupled(
    tr::Tree;
    Np::Integer = 1,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    proposal::Symbol = :local, guided::Bool = true, sorted::Bool = true,
    seed::UInt64 = rand(default_rng(),UInt64),
) = begin
    useguide = guided && tr.guided
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords); Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M); cll = zeros(M); zerofrac = zeros(M)
    ancestors = Vector{Int}(undef,Np)
    order = collect(1:Np)
    streams = [Stream(0,0) for _ ∈ 1:EVENTS, _ ∈ 1:Np]
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        initial!(tr,P,j,p,streams,seed;guided=useguide)
    end
    counts = zeros(Int,Np)
    for k ∈ k0:M
        propagate!(tr,P,W,k,p,streams,counts,seed;proposal,guided=useguide)
        nevents += sum(counts)
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
        k < M || break
        ## systematic resampling, in state order, with a uniform common to all parameters
        sorted && sortperm!(order, 1:Np; by = i -> statekey(P,i))
        systematic!(ancestors, W[order], rekey!(Stream(0,0), hash((seed,k,:resample))))
        for (j,a) ∈ enumerate(ancestors)
            copyparticle!(Q,j,P,order[a])
        end
        P, Q = Q, P
        W .= 1/Np
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

"""
    pfilter_ordered(tr; Np, params..., proposal, guided, threaded, rng)

FastSEIR's filter (Gillespie propagation, resampling after every event)
with the particles sorted by state before systematic resampling.
"""
pfilter_ordered(
    tr::Tree;
    Np::Integer = 1,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    proposal::Symbol = :local, guided::Bool = true, threaded::Bool = true,
    rng::AbstractRNG = default_rng(),
) = begin
    useguide = guided && tr.guided
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords); Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M); cll = zeros(M); zerofrac = zeros(M)
    ancestors = Vector{Int}(undef,Np)
    order = collect(1:Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        foreach_particle(P,seed,-j,threaded) do P, i, rng
            singular_part!(tr,P,i,j,p,rng;guided=useguide)
            0
        end
    end
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
        k < M || break
        sortperm!(order, 1:Np; by = i -> statekey(P,i))
        systematic!(ancestors, W[order], rng)
        for (j,a) ∈ enumerate(ancestors)
            copyparticle!(Q,j,P,order[a])
        end
        P, Q = Q, P
        W .= 1/Np
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

end
