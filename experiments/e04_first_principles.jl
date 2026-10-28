## E04: first-principles check of the likelihood and the simulator.
## P(no samples in [0,T]) and P(exactly one sample) estimated by forward simulation
## (SimSEIR) must equal, respectively, the exact likelihood of the empty genealogy and
## ∫ L(s) ds over genealogies with a single sample at time s (ExactSEIR).
include("common.jl")

const NSIM = parse(Int, get(ENV, "E04_NSIM", "200000"))
Random.seed!(404)

## Gauss–Legendre nodes on [0,1] (n = 24) via Golub–Welsch
using LinearAlgebra: SymTridiagonal, eigen
gauss_legendre(n) = begin
    b = [k/sqrt(4k^2-1) for k ∈ 1:n-1]
    F = eigen(SymTridiagonal(zeros(n), b))
    x = (F.values .+ 1) ./ 2
    w = F.vectors[1,:].^2
    x, w
end

single_sample_lik(s, T, par) = begin
    g = parse_newick("([&&PhyloPOMP type=sample]:$(s));", time=T)
    exp(ExactSEIR.loglik(g; par...))
end

empty_lik(T, par) = begin
    g = Genealogy{PhyloPOMP.Unstructured}(0.0, T)
    exp(ExactSEIR.loglik(g; par...))
end

cases = [
    (name="nondestructive", T=1.5, par=(β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.3, χ=0.0, pop=6, S0=3/6, E0=1/6, I0=1/6, R0=1/6)),
    (name="destructive",    T=1.5, par=(β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.2, χ=0.2, pop=6, S0=3/6, E0=1/6, I0=1/6, R0=1/6)),
    (name="larger_pop",     T=2.0, par=(β=2.5, σ=1.5, γ=0.8, ω=0.3, ψ=0.15, χ=0.0, pop=12, S0=8/12, E0=1/12, I0=2/12, R0=1/12)),
]

rows = NamedTuple[]
for c ∈ cases
    counts = zeros(Int, 3)            # 0, 1, ≥2 samples
    for _ ∈ 1:NSIM
        n = SimSEIR.simulate(; c.par..., time=c.T).nsample
        counts[min(n,2)+1] += 1
    end
    p̂ = counts ./ NSIM
    se = sqrt.(p̂ .* (1 .- p̂) ./ NSIM)
    x, w = gauss_legendre(24)
    p1 = c.T*sum(w[i]*single_sample_lik(c.T*x[i], c.T, c.par) for i ∈ eachindex(x))
    p0 = empty_lik(c.T, c.par)
    z0 = (p̂[1]-p0)/se[1]; z1 = (p̂[2]-p1)/se[2]
    @printf("%-15s P(0): sim=%.4f±%.4f exact=%.4f z=% .2f | P(1): sim=%.4f±%.4f exact=%.4f z=% .2f\n",
            c.name, p̂[1], se[1], p0, z0, p̂[2], se[2], p1, z1)
    push!(rows, (case=c.name, nsim=NSIM, p0_sim=p̂[1], p0_se=se[1], p0_exact=p0, z0=z0,
                 p1_sim=p̂[2], p1_se=se[2], p1_exact=p1, z1=z1))
end
write_csv(joinpath(RESULTS, "e04_first_principles.csv"), rows)
@assert all(r -> abs(r.z0) < 4 && abs(r.z1) < 4, rows) "first-principles check failed"
println("E04 PASS")
