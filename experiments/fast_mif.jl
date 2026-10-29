## Iterated filtering (IF2; Ionides et al. 2015) with FastSEIR, following the
## interface of POMP.jl's `mif`: each particle carries its own parameters,
## which are perturbed at lag 0 and after every genealogical event by
## `perturbations(scale, lag; params...)` (so `@perturbn` kernels work), with
## `scale = cooling(iteration)`, and resampled along with the particles.
module FastMif

using Random: AbstractRNG, default_rng
using Statistics: mean
using PhyloPOMP
using PhyloPOMP.FastSEIR: Tree, Particles, regular_part!, singular_part!,
    copyparticle!, systematic!, foreach_particle

geomean(x) = exp(mean(log, x))

rates(θ) = map(Float64, (; θ.β, θ.σ, θ.γ, θ.ω, θ.ψ, θ.χ, θ.pop))

perturb!(Θ, kernel, scale, lag) = begin
    for i ∈ eachindex(Θ)
        Θ[i] = merge(Θ[i], map(Float64, kernel(scale, lag; Θ[i]...)))
    end
    Θ
end

swarm_mean(Θ, avfun) = begin
    names = keys(first(Θ))
    NamedTuple{names}(map(n -> avfun(getfield.(Θ,n)), names))
end

"""
    mif(tr; Np, Nmif, perturbations, cooling, avfun = geomean, params,
        proposal = :local, guided = true, threaded = true, rng)

Returns `(; trace, logliks, params)`: the swarm means after each iteration
(`trace[1]` is the starting point), the log likelihood of each perturbed
filtering pass, and the final estimate.  `params` must contain every model
parameter of `FastSEIR.pfilter` (rates, `pop` and initial fractions).
"""
mif(
    tr::Tree;
    Np::Integer, Nmif::Integer,
    perturbations::Function, cooling::Function, avfun::Function = geomean,
    params::NamedTuple,
    proposal::Symbol = :local, guided::Bool = true, threaded::Bool = true,
    rng::AbstractRNG = default_rng(),
) = begin
    useguide = guided && tr.guided
    M = length(tr.time)
    k0 = findlast(==(tr.time[1]),tr.time)
    θ0 = map(Float64, params)
    Θ = fill(θ0,Np)
    Θnew = similar(Θ)
    P = Particles(Np,tr.nwords)
    Q = Particles(Np,tr.nwords)
    W = zeros(Np)
    ancestors = Vector{Int}(undef,Np)
    trace = [θ0]
    logliks = Float64[]
    for m ∈ 1:Nmif
        scale = cooling(m-1)
        perturb!(Θ,perturbations,scale,0)
        P.col .= 0; P.ellE .= 0; P.ellI .= 0; P.ll .= 0.0
        for i ∈ 1:Np
            θ = Θ[i]
            P.S[i], P.E[i], P.I[i], P.R[i] = barycentric(Int64,(θ.S0,θ.E0,θ.I0,θ.R0),Int(θ.pop))
        end
        seed = rand(rng,UInt64)
        for j ∈ 1:k0
            foreach_particle(P,seed,-j,threaded) do P, i, rng
                singular_part!(tr,P,i,j,rates(Θ[i]),rng;guided=useguide)
                0
            end
        end
        loglik = 0.0
        for k ∈ k0:M
            foreach_particle(P,seed,k,threaded) do P, i, rng
                isfinite(P.ll[i]) || return 0
                p = rates(Θ[i])
                regular_part!(tr,P,i,k,p,rng;proposal,guided=useguide)
                k < M && singular_part!(tr,P,i,k+1,p,rng;guided=useguide)
                0
            end
            llmax = maximum(P.ll)
            if !isfinite(llmax)
                loglik = -Inf
                break
            end
            W .= exp.(P.ll .- llmax)
            s = sum(W)
            loglik += llmax + log(s/Np)
            W ./= s
            systematic!(ancestors,W,rng)
            for (j,a) ∈ enumerate(ancestors)
                copyparticle!(Q,j,P,a)
                Θnew[j] = Θ[a]
            end
            P, Q = Q, P
            Θ .= Θnew
            (k < M || m < Nmif) && perturb!(Θ,perturbations,scale,k-k0+1)
        end
        push!(logliks,loglik)
        push!(trace,swarm_mean(Θ,avfun))
    end
    (; trace, logliks, params = trace[end])
end

end
