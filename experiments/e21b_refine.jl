## E21b: refine the seir_trees[2] estimate of E21 (smaller random walk, more
## particles) and measure the filter's precision there as Np varies.
include("common.jl")
include("fast_mif.jl")
using .FastMif: mif

const NMIF = parse(Int, get(ENV, "E21B_NMIF", "30"))
const NP = parse(Int, get(ENV, "E21B_NP", "4000"))
const R = parse(Int, get(ENV, "E21B_R", "8"))
const RW = 0.002
const RWPOP = 0.05

g = parse_newick(FastSEIR.seir_trees[2])
tr = FastSEIR.Tree(g, default_guide())

## best E21 endpoint (start "pop×2")
fits = filter(r -> r.run == "mif", let lines = readlines(joinpath(RESULTS, "e21_tree2_mif.csv"))
    hdr = Symbol.(split(lines[1], ","))
    [NamedTuple{Tuple(hdr)}(Tuple(map(x -> something(tryparse(Float64, x), strip(x, '"')), split(l, ",")))) for l ∈ lines[2:end]]
end)
best = argmax(r -> r.mean, fits)
θ0 = merge(DEFAULTS, (β=best.β, σ=best.σ, γ=best.γ, ω=best.ω, ψ=best.ψ, pop=round(best.pop)))
@printf("start (E21 %s): β=%.3f σ=%.3f γ=%.3f ω=%.3f ψ=%.4f pop=%d\n",
        best.start, θ0.β, θ0.σ, θ0.γ, θ0.ω, θ0.ψ, θ0.pop)

kernel(scale, lag; β, σ, γ, ω, ψ, pop, _...) = begin
    s = scale*RW
    (β = β*exp(s*randn()), σ = σ*exp(s*randn()), γ = γ*exp(s*randn()),
     ω = ω*exp(s*randn()), ψ = ψ*exp(s*randn()),
     pop = lag == 0 ? max(20.0, round(pop*exp(scale*RWPOP*randn()))) : pop)
end
cooling(m) = 0.5^(m/50)

Random.seed!(2122)
t = @elapsed fit = mif(tr; Np=NP, Nmif=NMIF, perturbations=kernel, cooling, params=θ0, rng=Xoshiro(2122))
θ = merge(fit.params, (pop = round(Int, fit.params.pop),))
@printf("refined (IF2 %.0fs): β=%.3f σ=%.3f γ=%.3f ω=%.3f ψ=%.4f pop=%d\n",
        t, θ.β, θ.σ, θ.γ, θ.ω, θ.ψ, θ.pop)
write_csv(joinpath(RESULTS, "e21b_trace.csv"),
          [(iteration=i-1, β=p.β, σ=p.σ, γ=p.γ, ω=p.ω, ψ=p.ψ, pop=p.pop,
            loglik=(i > 1 ? fit.logliks[i-1] : NaN)) for (i, p) ∈ enumerate(fit.trace)])

rows = NamedTuple[]
for (label, par) ∈ (("estimate", θ), ("defaults", DEFAULTS)), Np ∈ (1000, 4000, 16000)
    label == "defaults" && Np ≠ 1000 && continue     # defaults at 4000/16000: E18/E20
    pfilter(tr; Np=50, par...)
    res = [(@timed logLik(pfilter(tr; Np, par...))) for _ ∈ 1:R]
    ll = [r.value for r ∈ res]
    s = summarize(ll)
    @printf("%-8s Np=%5d  mean=%.2f  sd=%.2f  lme=%.2f  t=%.1fs\n", label, Np, mean(ll), s.sd, s.lme, mean(r.time for r ∈ res))
    push!(rows, (params=label, Np=Np, R=R, mean=mean(ll), sd=s.sd, lme=s.lme, sec=mean(r.time for r ∈ res),
                 β=par.β, σ=par.σ, γ=par.γ, ω=par.ω, ψ=par.ψ, pop=Float64(par.pop)))
    write_csv(joinpath(RESULTS, "e21b_precision.csv"), rows)
end
