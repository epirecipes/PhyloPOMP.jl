## Nested SMC: each outer particle carries one population and K conditional
## colour particles. Population recoveries are unrestricted; incompatible
## colours receive zero weight instead of altering the shared population rate.
module NestedColour

using Random: AbstractRNG, Xoshiro, default_rng, randexp
using Base.Threads: @threads, nthreads
using PhyloPOMP
using PhyloPOMP: Node
import PhyloPOMP: logLik, eff_sample_size, cond_logLik
const F = PhyloPOMP.FastSEIR

struct NestedPfilterd
    logLik::Float64
    eff_sample_size::Vector{Float64}
    cond_logLik::Vector{Float64}
    nevents::Int
    inner_resamples::Int
end

logLik(pf::NestedPfilterd) = pf.logLik
eff_sample_size(pf::NestedPfilterd) = pf.eff_sample_size
cond_logLik(pf::NestedPfilterd) = pf.cond_logLik

struct Workspace
    weights::Vector{Float64}
    ancestors::Vector{Int}
    sumE::Vector{Float64}
    sumI::Vector{Float64}
    tmpE::Vector{Float64}
    tmpI::Vector{Float64}
end

Workspace(K) = Workspace(zeros(K),zeros(Int,K),zeros(K),zeros(K),zeros(K),zeros(K))

## Return the log normalizer only when resampling, or at a forced boundary.
## Until then, unnormalized colour weights remain in P.ll.
settle!(P, C, lo, w, rng, trigger; force=false) = begin
    K = length(w.weights)
    hi = lo+K-1
    mx = maximum(@view P.ll[lo:hi])
    isfinite(mx) || return (-Inf,0)
    if K == 1
        force || return (0.0,0)
        P.ll[lo] = 0.0
        return (mx,0)
    end
    for j ∈ 1:K
        w.weights[j] = exp(P.ll[lo+j-1]-mx)
    end
    total = sum(w.weights)
    w.weights ./= total
    force || 1/sum(abs2,w.weights) ≤ trigger*K || return (0.0,0)
    F.systematic!(w.ancestors,w.weights,rng)
    for (j,a) ∈ enumerate(w.ancestors)
        F.copyparticle!(C,lo+j-1,P,lo+a-1)
        w.tmpE[j] = w.sumE[a]
        w.tmpI[j] = w.sumI[a]
    end
    for j ∈ 1:K
        F.copyparticle!(P,lo+j-1,C,lo+j-1)
    end
    w.sumE .= w.tmpE
    w.sumI .= w.tmpI
    (mx+log(total/K),1)
end

singular_group!(tr, P, lo, K, k, p, rng, guided) = begin
    S, E, I, R = P.S[lo], P.E[lo], P.I[lo], P.R[lo]
    for i ∈ lo:lo+K-1
        isfinite(P.ll[i]) || continue
        F.singular_part!(tr,P,i,k,p,rng; guided)
    end
    ## A forced branch changes the common population even when some colour
    ## particles were incompatible and returned before that update.
    if tr.type[k] == Node && S > 0
        S -= 1
        E += 1
    end
    for i ∈ lo:lo+K-1
        P.S[i] = S; P.E[i] = E; P.I[i] = I; P.R[i] = R
    end
    nothing
end

regular_group!(tr, P, C, lo, k, p, w, rng, guided, trigger, check) = begin
    K = length(w.weights)
    S, E, I, R = P.S[lo], P.E[lo], P.I[lo], P.R[lo]
    t, tf = tr.time[k], tr.tend[k]
    ll, nr = settle!(P,C,lo,w,rng,trigger)
    isfinite(ll) || return (ll,0,nr)
    for j ∈ 1:K
        w.sumE[j], w.sumI[j] = F.ratio_sums(tr,P,lo+j-1,k)
    end
    nevents = 0
    while true
        transmission = p.β*S*I/p.pop
        progression = p.σ*E
        recovery = p.γ*I
        waning = p.ω*R
        rate = transmission+progression+recovery+waning
        step = randexp(rng)/rate
        done = t+step ≥ tf
        ll -= p.ψ*I*(done ? tf-t : step)
        done && break
        t += step
        nevents += 1
        u = rand(rng)*rate
        if u < transmission+progression
            toI = u ≥ transmission
            for j ∈ 1:K
                i = lo+j-1
                isfinite(P.ll[i]) || continue
                P.E[i] = E; P.I[i] = I
                b, lw = F.color_change(tr,P,i,k,t,toI,w.sumE[j],w.sumI[j],rng;
                                       proposal=:local,guided)
                P.ll[i] += lw
                if b > 0
                    if toI
                        F.infec!(P,i,tr.lins[b])
                        P.ellE[i] -= 1; P.ellI[i] += 1
                    else
                        F.expos!(P,i,tr.lins[b])
                        P.ellI[i] -= 1; P.ellE[i] += 1
                    end
                    if b ≠ tr.focal[k]
                        w.sumE[j] += toI ? -tr.toI[b] : tr.toI[b]
                        w.sumI[j] += toI ? tr.toE[b] : -tr.toE[b]
                    end
                end
            end
            if toI
                E -= 1; I += 1
            else
                S -= 1; E += 1
            end
        elseif u < transmission+progression+recovery
            for i ∈ lo:lo+K-1
                I ≤ P.ellI[i] && (P.ll[i] = -Inf)
            end
            I -= 1; R += 1
        else
            R -= 1; S += 1
        end
        if nevents % check == 0
            gain, n = settle!(P,C,lo,w,rng,trigger)
            ll += gain
            nr += n
            isfinite(ll) || return (ll,nevents,nr)
        end
    end
    for i ∈ lo:lo+K-1
        P.S[i] = S; P.E[i] = E; P.I[i] = I; P.R[i] = R
    end
    (ll,nevents,nr)
end

propagate_groups!(tr, P, C, k, k0, p, ll, seed, Np, K, guided, trigger, check, threaded) = begin
    nchunk = min(64,Np)
    counts = zeros(Int,nchunk,2)
    work(c) = begin
        stream = Xoshiro(hash((seed,k,c)))
        w = Workspace(K)
        for a ∈ c:nchunk:Np
            lo = (a-1)*K+1
            if k == k0
                for j ∈ 1:k0
                    singular_group!(tr,P,lo,K,j,p,stream,guided)
                end
            end
            gain, n, nr = regular_group!(tr,P,C,lo,k,p,w,stream,guided,trigger,check)
            counts[c,1] += n
            counts[c,2] += nr
            if isfinite(gain)
                k < length(tr.time) && singular_group!(tr,P,lo,K,k+1,p,stream,guided)
                tail, nr = settle!(P,C,lo,w,stream,trigger; force=true)
                gain += tail
                counts[c,2] += nr
            end
            ll[a] = gain
        end
    end
    if threaded && nthreads() > 1
        @threads :static for c ∈ 1:nchunk
            work(c)
        end
    else
        foreach(work,1:nchunk)
    end
    (sum(@view counts[:,1]),sum(@view counts[:,2]))
end

"""
    pfilter_nested(tr; Np=1, K=4, inner_trigger=0.5, inner_check=32, params...)

Experimental nested particle filter with `Np` population particles and `K`
conditional colour particles per population. The full population Gillespie rates
are shared across colours. Incompatible recoveries kill colour particles; there
is no additional recovery killing potential. Between observed events, resample
colours when their ESS falls below `inner_trigger*K`, checked every `inner_check`
population events. At observed-event boundaries, resample colours and then
populations, carrying both levels' normalizing constants into the likelihood.

This is finite inner SMC, not exact analytic colour integration or a product
closure. It retains the exact likelihood target. Only non-destructive sampling
(`χ=0`) and the local colour proposal are supported. It does not estimate scores.
"""
pfilter_nested(
    tr::F.Tree;
    Np::Integer=1, K::Integer=4,
    inner_trigger::Real=0.5, inner_check::Integer=32,
    β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
    pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08,
    guided::Bool=true, threaded::Bool=true, rng::AbstractRNG=default_rng(),
) = begin
    @assert Np > 0 && K > 0 && inner_check > 0
    @assert 0 ≤ inner_trigger ≤ 1
    @assert χ == 0 "nested colours currently require non-destructive sampling (χ=0)"
    p = map(Float64,(;β,σ,γ,ω,ψ,χ,pop))
    guided = guided && tr.guided
    P = F.Particles(Np*K,tr.nwords)
    Q = F.Particles(Np*K,tr.nwords)
    C = F.Particles(Np*K,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    M = length(tr.time)
    k0 = findlast(==(tr.time[1]),tr.time)
    ess = fill(Float64(Np),M)
    cll = zeros(M)
    ll = zeros(Np)
    W = zeros(Np)
    ancestors = zeros(Int,Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = inner_resamples = 0
    for k ∈ k0:M
        n, nr = propagate_groups!(tr,P,C,k,k0,p,ll,seed,Np,K,guided,inner_trigger,inner_check,threaded)
        nevents += n
        inner_resamples += nr
        mx = maximum(ll)
        isfinite(mx) || return NestedPfilterd(-Inf,ess,cll,nevents,inner_resamples)
        W .= exp.(ll .- mx)
        total = sum(W)
        cll[k] = mx+log(total/Np)
        loglik += cll[k]
        W ./= total
        ess[k] = 1/sum(abs2,W)
        F.systematic!(ancestors,W,rng)
        for (j,a) ∈ enumerate(ancestors), b ∈ 1:K
            F.copyparticle!(Q,(j-1)*K+b,P,(a-1)*K+b)
        end
        P, Q = Q, P
    end
    NestedPfilterd(loglik,ess,cll,nevents,inner_resamples)
end

end
