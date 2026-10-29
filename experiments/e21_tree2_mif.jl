## E21: is seir_trees[2]'s degeneracy model misfit?  Its sampling rate
## (864 samples over ≈400 time units, ≈2.2 per unit time) is about four times
## what the default parameters predict (ψ I* ≈ 0.5).  Estimate the rates and the
## population size by IF2 (experiments/fast_mif.jl), then compare the filter's
## precision at the estimate with that at the defaults.
include("common.jl")
include("fast_mif.jl")
using .FastMif: mif

const NMIF = parse(Int, get(ENV, "E21_NMIF", "40"))
const NP = parse(Int, get(ENV, "E21_NP", "2000"))
const NPEVAL = parse(Int, get(ENV, "E21_NPEVAL", "4000"))
const REVAL = parse(Int, get(ENV, "E21_REVAL", "8"))
const RW = 0.005            # per event; ≈1560 events per pass
const RWPOP = 0.1           # population size: perturbed at lag 0 only

g = parse_newick(FastSEIR.seir_trees[2])
tr = FastSEIR.Tree(g, default_guide())
println("seir_trees[2]: $(g.nsample) samples, $(length(tr.time)) events, T = $(round(tr.tend[end], digits=1))")

kernel(scale, lag; β, σ, γ, ω, ψ, pop, _...) = begin
    s = scale*RW
    (β = β*exp(s*randn()), σ = σ*exp(s*randn()), γ = γ*exp(s*randn()),
     ω = ω*exp(s*randn()), ψ = ψ*exp(s*randn()),
     pop = lag == 0 ? max(20.0, round(pop*exp(scale*RWPOP*randn()))) : pop)
end
cooling(m) = 0.5^(m/50)

evaluate(θ) = begin
    θ = merge(θ, (pop = round(Int, θ.pop),))
    pfilter(tr; Np=50, θ...)
    res = [(@timed logLik(pfilter(tr; Np=NPEVAL, θ...))) for _ ∈ 1:REVAL]
    ll = [r.value for r ∈ res]
    (; mean = mean(ll), sd = std(ll), lme = logmeanexp(ll), sec = mean(r.time for r ∈ res))
end

Random.seed!(2121)
starts = [
    ("defaults", DEFAULTS),
    ("ψ×4", merge(DEFAULTS, (ψ = 0.08,))),
    ("pop×2", merge(DEFAULTS, (pop = 200,))),
]
rows = NamedTuple[]
e = evaluate(DEFAULTS)
@printf("%-9s β=%.2f σ=%.2f γ=%.2f ω=%.2f ψ=%.4f pop=%d  mean=%.1f sd=%.2f lme=%.1f [%.1fs/run]\n",
        "defaults", DEFAULTS.β, DEFAULTS.σ, DEFAULTS.γ, DEFAULTS.ω, DEFAULTS.ψ, DEFAULTS.pop,
        e.mean, e.sd, e.lme, e.sec)
push!(rows, (run="defaults", start="", β=DEFAULTS.β, σ=DEFAULTS.σ, γ=DEFAULTS.γ, ω=DEFAULTS.ω,
             ψ=DEFAULTS.ψ, pop=Float64(DEFAULTS.pop), mean=e.mean, sd=e.sd, lme=e.lme, sec=e.sec, mif_sec=0.0))
for (label, θ0) ∈ starts
    t = @elapsed fit = mif(tr; Np=NP, Nmif=NMIF, perturbations=kernel, cooling,
                           params=θ0, rng=Xoshiro(hash(label)))
    θ = fit.params
    ev = evaluate(θ)
    @printf("mif from %-9s → β=%.2f σ=%.2f γ=%.2f ω=%.2f ψ=%.4f pop=%.0f  mean=%.1f sd=%.2f lme=%.1f [%.1fs/run; IF2 %.0fs]\n",
            label, θ.β, θ.σ, θ.γ, θ.ω, θ.ψ, θ.pop, ev.mean, ev.sd, ev.lme, ev.sec, t)
    push!(rows, (run="mif", start=label, β=θ.β, σ=θ.σ, γ=θ.γ, ω=θ.ω, ψ=θ.ψ, pop=θ.pop,
                 mean=ev.mean, sd=ev.sd, lme=ev.lme, sec=ev.sec, mif_sec=t))
    write_csv(joinpath(RESULTS, "e21_tree2_mif_trace_$(replace(label, "×"=>"x", "ψ"=>"psi")).csv"),
              [(iteration=i-1, β=p.β, σ=p.σ, γ=p.γ, ω=p.ω, ψ=p.ψ, pop=p.pop,
                loglik=(i > 1 ? fit.logliks[i-1] : NaN)) for (i, p) ∈ enumerate(fit.trace)])
    write_csv(joinpath(RESULTS, "e21_tree2_mif.csv"), rows)
end
