module ResamplingFilter

using Random: AbstractRNG, default_rng
using PhyloPOMP
const F = PhyloPOMP.FastSEIR

stratified!(ancestors, weights, rng) = begin
    N = length(ancestors)
    c, j = weights[1], 1
    for i ∈ eachindex(ancestors)
        u = (i-1+rand(rng))/N
        while u ≥ c && j < length(weights)
            j += 1
            c += weights[j]
        end
        ancestors[i] = j
    end
    ancestors
end

residual_stratified!(ancestors, weights, residual, rng) = begin
    N = length(ancestors)
    used = 0
    for i ∈ eachindex(weights)
        expected = N*weights[i]
        copies = floor(Int,expected)
        residual[i] = expected-copies
        for _ ∈ 1:copies
            used += 1
            ancestors[used] = i
        end
    end
    @assert used ≤ N "invalid normalized weights"
    if used < N
        total = sum(residual)
        @assert total > 0 "zero residual mass with unfilled offspring"
        residual ./= total
        stratified!(@view(ancestors[used+1:N]),residual,rng)
    end
    ancestors
end

resample!(ancestors, weights, residual, method, rng) = if method == :systematic
    F.systematic!(ancestors,weights,rng)
elseif method == :stratified
    stratified!(ancestors,weights,rng)
elseif method == :residual
    residual_stratified!(ancestors,weights,residual,rng)
else
    error("unknown resampling method '$method'")
end

"""
    pfilter_resampling(tr; Np=1, resampling=:systematic, model_params...,
                       guided=true, threaded=true, rng)

FastSEIR's exact every-event filter with systematic, stratified, or
residual-stratified ancestor selection. Only resampling changes: all three
schemes have expected offspring counts Np*W. The systematic research control
reproduces the canonical filter at the same seed. Returns FastPfilterd.
"""
pfilter_resampling(
    tr::F.Tree; Np::Integer=1, resampling::Symbol=:systematic,
    β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
    pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08,
    guided::Bool=true, threaded::Bool=true,
    rng::AbstractRNG=default_rng(),
) = begin
    @assert Np > 0 && resampling ∈ (:systematic,:stratified,:residual)
    useguide = guided && tr.guided
    p = map(Float64,(;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = F.Particles(Np,tr.nwords)
    Q = F.Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M)
    cll = zeros(M)
    zerofrac = zeros(M)
    ancestors = zeros(Int,Np)
    residual = zeros(Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        F.foreach_particle(P,seed,-j,threaded) do P, i, rng
            F.singular_part!(tr,P,i,j,p,rng; guided=useguide)
            0
        end
    end
    for k ∈ k0:M
        nevents += F.foreach_particle(P,seed,k,threaded) do P, i, rng
            (W[i] > 0 && isfinite(P.ll[i])) || return 0
            n = F.regular_part!(tr,P,i,k,p,rng; proposal=:local,guided=useguide)
            k < M && F.singular_part!(tr,P,i,k+1,p,rng; guided=useguide)
            n
        end
        mx = maximum(P.ll)
        isfinite(mx) || return F.FastPfilterd(-Inf,ess,cll,zerofrac,nevents)
        W .*= exp.(P.ll .- mx)
        total = sum(W)
        cll[k] = mx+log(total)
        loglik += cll[k]
        W ./= total
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/Np
        resample!(ancestors,W,residual,resampling,rng)
        for (j,a) ∈ enumerate(ancestors)
            F.copyparticle!(Q,j,P,a)
        end
        P, Q = Q, P
        W .= 1/Np
    end
    F.FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

end
