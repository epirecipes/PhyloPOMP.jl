## Experimental auxiliary particle filter built on FastSEIR's internals.
##
## After each genealogical event the particles are selected according to a
## look-ahead to the *next* event, evaluated at their current state:
##   log ĝ = -(ψ+χ) I Δt                  (sampling decay over the next interval)
##         + log w(x)                     (weight of the next event at the current state)
##         + log h_b(c_b) - log h_b(I)    (guide compatibility of the event's parent lineage)
## raised to the power `temper`.  The selection is corrected exactly (the next
## increment is divided by ĝ of the ancestor), so the likelihood target is unchanged.
module APF

using Random: AbstractRNG, default_rng
using PhyloPOMP
using PhyloPOMP: Root, Node, Sample
using PhyloPOMP.FastSEIR: Tree, Particles, FastPfilterd, regular_part!, singular_part!,
    copyparticle!, systematic!, foreach_particle, isinfec, guideprob

## these filters do not estimate the sufficient statistics
const NOSTATS = PhyloPOMP.FastSEIR.SuffStats(ntuple(_->NaN,length(PhyloPOMP.FastSEIR.STATS)))

## log look-ahead for particle i, which is at the start of interval k (k < M)
lookahead(tr::Tree, P::Particles, i, k, p, guided::Bool) = begin
    (; β, γ, ψ, χ, pop) = p
    S, E, I, ellE, ellI = P.S[i], P.E[i], P.I[i], P.ellE[i], P.ellI[i]
    Δt = tr.tend[k] - tr.time[k]
    s = -((ψ+χ)*I + (I ≤ ellI ? γ*I : 0.0))*Δt
    k < length(tr.time) || return s      # the last interval has no closing event
    e = k+1
    b = tr.parlin[e]
    if tr.type[e] == Sample && isempty(tr.chillins[e])
        s += log(ψ*(I-ellI+1) + χ*I)
    elseif tr.type[e] == Sample
        s += log(ψ)
    elseif tr.type[e] == Node
        s += log(β*max(S,0.5)/(pop*(E+1)))
    end
    ## the parent lineage must be infectious at the event: guide backward
    ## messages h(E), h(I) at the current time, relative to the event's target
    if guided
        j = findfirst(==(b), @view(tr.lins[tr.ptr[k]:tr.ptr[k+1]-1])) + tr.ptr[k] - 1
        hE, hI = guideprob(tr, j, exp(-tr.κ*Δt))
        s += isinfec(P,i,b) ? 0.0 : log(max(hE/hI, 1e-12))
    else
        s += isinfec(P,i,b) ? 0.0 : log(0.1)
    end
    s
end

pfilter_apf(
    tr::Tree;
    Np::Integer = 1, temper::Real = 1.0,
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
    lg = zeros(Np); Waux = zeros(Np)
    ancestors = Vector{Int}(undef,Np)
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
        isfinite(llmax) || return FastPfilterd(-Inf,ess,cll,zerofrac,nevents,NOSTATS)
        W .*= exp.(P.ll .- llmax)
        s = sum(W)
        cll[k] = llmax+log(s)
        loglik += cll[k]
        W ./= s
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/Np
        k < M || break
        ## auxiliary selection for the next step (interval k+1, event k+2)
        for i ∈ 1:Np
            lg[i] = W[i] > 0 ? temper*lookahead(tr,P,i,k+1,p,useguide) : -Inf
        end
        lgmax = maximum(lg)
        Waux .= W .* exp.(lg .- lgmax)
        aux = sum(Waux)
        loglik += log(aux)      # the shift lgmax cancels with the correction below
        Waux ./= aux
        systematic!(ancestors,Waux,rng)
        for (j,a) ∈ enumerate(ancestors)
            copyparticle!(Q,j,P,a)
            Q.ll[j] = -(lg[a]-lgmax)
        end
        P, Q = Q, P
        W .= 1/Np
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents,NOSTATS)
end

end
