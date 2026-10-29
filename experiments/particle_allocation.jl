module ParticleAllocation

using Random: AbstractRNG, default_rng
using PhyloPOMP
const F = PhyloPOMP.FastSEIR

resample!(
    ancestors, weights, rng,
) = begin
    length(ancestors) == length(weights) &&
        return F.systematic!(ancestors,weights,rng)
    m = length(ancestors)
    u = rand(rng)/m
    c, j = weights[1], 1
    for i ∈ eachindex(ancestors)
        while u > c && j < length(weights)
            j += 1
            c += weights[j]
        end
        ancestors[i] = j
        u += 1/m
    end
    ancestors
end

"""
    pfilter_allocated(tr, sizes; model_params..., guided=true, threaded=true, rng)

Exact FastSEIR kernels with a predetermined number of particles per interval.
The schedule must be chosen independently of this evaluation's random draws.
After weighting at interval k, systematic resampling produces sizes[k+1]
ancestors. Each likelihood increment uses the current, not next, particle
count. A constant schedule reproduces the standard every-event filter.

Returns `(pf, events)`, including population-event counts per interval for
independent pilot planning. Only two particle clouds are retained; changing
dimensions reallocates the destination cloud.
"""
pfilter_allocated(
    tr::F.Tree, sizes::AbstractVector{<:Integer};
    β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
    pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08,
    guided::Bool=true, threaded::Bool=true,
    rng::AbstractRNG=default_rng(),
) = begin
    M = length(tr.time)
    @assert length(sizes) == M && all(>(0),sizes)
    useguide = guided && tr.guided
    p = map(Float64,(;β,σ,γ,ω,ψ,χ,pop))
    k0 = findlast(==(tr.time[1]),tr.time)
    N = sizes[k0]
    P = F.Particles(N,tr.nwords)
    Q = F.Particles(N,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/N,N)
    ess = Float64.(sizes)
    cll = zeros(M)
    zerofrac = zeros(M)
    events = zeros(Int,M)
    ancestors = zeros(Int,N)
    seed = rand(rng,UInt64)
    loglik = 0.0
    for j ∈ 1:k0
        F.foreach_particle(P,seed,-j,threaded) do P, i, rng
            F.singular_part!(tr,P,i,j,p,rng; guided=useguide)
            0
        end
    end
    for k ∈ k0:M
        events[k] = F.foreach_particle(P,seed,k,threaded) do P, i, rng
            (W[i] > 0 && isfinite(P.ll[i])) || return 0
            n = F.regular_part!(tr,P,i,k,p,rng; proposal=:local,guided=useguide)
            k < M && F.singular_part!(tr,P,i,k+1,p,rng; guided=useguide)
            n
        end
        mx = maximum(P.ll)
        if !isfinite(mx)
            pf = F.FastPfilterd(-Inf,ess,cll,zerofrac,sum(events))
            return (;pf,events)
        end
        W .*= exp.(P.ll .- mx)
        total = sum(W)
        cll[k] = mx+log(total)
        loglik += cll[k]
        W ./= total
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/sizes[k]
        nextN = k < M ? sizes[k+1] : sizes[k]
        length(Q.ll) == nextN || (Q = F.Particles(nextN,tr.nwords))
        resize!(ancestors,nextN)
        resample!(ancestors,W,rng)
        for (j,a) ∈ enumerate(ancestors)
            F.copyparticle!(Q,j,P,a)
        end
        P, Q = Q, P
        if length(W) == nextN
            W .= 1/nextN
        else
            W = fill(1/nextN,nextN)
        end
    end
    pf = F.FastPfilterd(loglik,ess,cll,zerofrac,sum(events))
    (;pf,events)
end

"""
    allocate(variance, cost, uniform; lower, upper, quantum=16)

Minimize the pilot proxy sum(variance[k]/N[k]) at the uniform schedule's
predicted work sum(uniform*cost[k]). Continuous optimum is proportional to
sqrt(variance/cost), clipped to bounds, then rounded to quantum. This is a
planning heuristic, not an estimate of the full likelihood's variance.
"""
allocate(
    variance, cost, uniform::Integer;
    lower::Integer, upper::Integer, quantum::Integer=16,
) = begin
    @assert length(variance) == length(cost) > 0
    @assert all(isfinite,variance) && all(≥(0),variance) && any(>(0),variance)
    @assert all(isfinite,cost) && all(>(0),cost)
    @assert 0 < lower ≤ uniform ≤ upper && quantum > 0
    @assert lower % quantum == upper % quantum == 0
    target = uniform*sum(cost)
    scale = sqrt.(variance ./ cost)
    work(λ) = sum(clamp.(λ .* scale,lower,upper) .* cost)
    ## Zero-variance intervals remain at the floor even at infinite scale.
    @assert sum((variance .> 0) .* upper .* cost .+
                (variance .== 0) .* lower .* cost) ≥ target "unattainable pilot work budget"
    lo, hi = 0.0, 1.0
    while work(hi) < target
        hi *= 2
        @assert isfinite(hi) "allocation scale overflow"
    end
    for _ ∈ 1:80
        mid = (lo+hi)/2
        work(mid) < target ? (lo = mid) : (hi = mid)
    end
    clamp.(round.(Int,clamp.(hi .* scale,lower,upper) ./ quantum) .* quantum,lower,upper)
end

end
