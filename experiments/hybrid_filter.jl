## Approximate hybrid filter for large populations (τ-leaping).
##
## The genealogical events are handled exactly, as in FastSEIR, but between
## them the population process is advanced in steps of length at most `h`:
## in each step the numbers of transmissions, progressions, recoveries and
## losses of immunity are Poisson with the rates at the start of the step,
## each transmission (progression) moves a lineage from I to E (E to I) with
## the locally optimal probability at the counts at the start of the step,
## and the weight is multiplied by the corresponding totals Σφ of the exact
## color weights.  The cost of a step does not depend on the population
## size, but the result approximates the likelihood with an error that
## vanishes as h → 0.  Proposals are not guided.
module Hybrid

using Random: AbstractRNG, default_rng
using Distributions: Poisson, Binomial
using PhyloPOMP
using PhyloPOMP.FastSEIR: Tree, Particles, FastPfilterd, isinfec, infec!, expos!,
    singular_part!, copyparticle!, systematic!, foreach_particle

poisson(rng, λ) = λ > 0 ? rand(rng, Poisson(λ)) : 0
binomial(rng, n, p) = n > 0 && p > 0 ? rand(rng, Binomial(n, min(p,1.0))) : 0

## Advance particle `i` over interval `k` in τ-leaps of length at most `h`.
## Returns the number of population events.
leap_part!(tr::Tree, P::Particles, i, k, p, rng; h) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    S, E, I, R = P.S[i], P.E[i], P.I[i], P.R[i]
    ellE, ellI = P.ellE[i], P.ellI[i]
    ll = P.ll[i]
    nevent = 0
    while t < tf
        Δ = min(h, tf-t)
        nT = poisson(rng, β*S*I/pop*Δ)
        nP = poisson(rng, σ*E*Δ)
        nR = I > ellI ? poisson(rng, γ*I*Δ) : 0
        nW = poisson(rng, ω*R*Δ)
        ll -= ((ψ+χ)*I + @indicator(I ≤ ellI, γ*I))*Δ
        ## color changes at the counts at the start of the step
        KT = KP = 0
        if nT > 0
            φ0, φb = (E+1-ellE)/(E+1), (I+1-ellI)/((E+1)*I)
            Σ = φ0 + ellI*φb
            ll += nT*log(Σ)
            KT = min(binomial(rng, nT, ellI*φb/Σ), ellI)
        end
        if nP > 0
            φ0, φb = @indicator(E > ellE, (I+1-ellI)/(I+1)), 1/(I+1)
            Σ = φ0 + ellE*φb
            ll += nP*log(Σ)
            KP = min(binomial(rng, nP, ellE*φb/Σ), ellE)
        end
        ## choose the lineages that move uniformly (selection sampling), from
        ## their demes at the start of the step
        if KT+KP > 0
            mI, mE = ellI, ellE
            for j ∈ tr.ptr[k]:tr.ptr[k+1]-1
                b = tr.lins[j]
                if isinfec(P,i,b)
                    if rand(rng)*mI < KT
                        expos!(P,i,b); KT -= 1; ellI -= 1; ellE += 1
                    end
                    mI -= 1
                else
                    if rand(rng)*mE < KP
                        infec!(P,i,b); KP -= 1; ellE -= 1; ellI += 1
                    end
                    mE -= 1
                end
            end
        end
        S += nW-nT; E += nT-nP; I += nP-nR; R += nR-nW
        nevent += nT+nP+nR+nW
        if S < 0 || R < 0 || E < ellE || I < ellI
            ll = -Inf
            break
        end
        t += Δ
    end
    P.S[i], P.E[i], P.I[i], P.R[i] = S, E, I, R
    P.ellE[i], P.ellI[i] = ellE, ellI
    P.ll[i] = ll
    nevent
end

"""
    pfilter_hybrid(tr; Np, h, params..., threaded, rng)

FastSEIR's filter (resampling after every event, unguided proposals) with
τ-leaping of step `h` between genealogical events.
"""
pfilter_hybrid(
    tr::Tree;
    Np::Integer = 1, h::Real = 0.05,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    threaded::Bool = true, rng::AbstractRNG = default_rng(),
) = begin
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords); Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M); cll = zeros(M); zerofrac = zeros(M)
    ancestors = Vector{Int}(undef,Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        foreach_particle(P,seed,-j,threaded) do P, i, rng
            singular_part!(tr,P,i,j,p,rng;guided=false)
            0
        end
    end
    for k ∈ k0:M
        nevents += foreach_particle(P,seed,k,threaded) do P, i, rng
            (W[i] > 0 && isfinite(P.ll[i])) || return 0
            n = leap_part!(tr,P,i,k,p,rng;h)
            k < M && isfinite(P.ll[i]) && singular_part!(tr,P,i,k+1,p,rng;guided=false)
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
        systematic!(ancestors,W,rng)
        for (j,a) ∈ enumerate(ancestors)
            copyparticle!(Q,j,P,a)
        end
        P, Q = Q, P
        W .= 1/Np
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

end
