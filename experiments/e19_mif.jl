## E19: parameter estimation by iterated filtering (IF2) with FastSEIR
## (experiments/fast_mif.jl) on a simulated genealogy with known parameters.
## Estimates β, γ, ψ from random starts; σ, ω and the initial state are fixed.
include("common.jl")
include("fast_mif.jl")
using .FastMif: mif

const NMIF = parse(Int, get(ENV, "E19_NMIF", "60"))
const NSTART = parse(Int, get(ENV, "E19_NSTART", "8"))
const NP = parse(Int, get(ENV, "E19_NP", "1000"))
const RW = 0.02
const PARTS = get(ENV, "E19_PARTS", "AB")

truth = merge(DEFAULTS, (ψ=0.05,))
g = let rng = Xoshiro(1919), sim = nothing
    while true
        sim = SimSEIR.simulate(; truth..., time=60.0, rng)
        80 ≤ sim.genealogy.nsample ≤ 160 && break
    end
    sim.genealogy
end
tr = FastSEIR.Tree(g, default_guide())
println("simulated genealogy: $(g.nsample) samples, $(length(tr.time)) events")

kernel(scale, lag; β, γ, ψ, _...) = (
    β = β*exp(scale*RW*randn()),
    γ = γ*exp(scale*RW*randn()),
    ψ = ψ*exp(scale*RW*randn()),
)
cooling(m) = 0.5^(m/50)

evaluate(θ; R=10, Np=2000) = summarize([logLik(pfilter(tr; Np, θ..., pop=round(Int, θ.pop))) for _ ∈ 1:R])

## ---------------- Part A: IF2 from random starts
if occursin('A', PARTS)
Random.seed!(19)
starts = [merge(truth, (β=2*4^rand(), γ=0.5*4^rand(), ψ=0.02*5^rand())) for _ ∈ 1:NSTART]

rows = NamedTuple[]
et = evaluate(truth)
@printf("truth           β=%.3f γ=%.3f ψ=%.4f  logLik=%.2f ± %.2f\n", truth.β, truth.γ, truth.ψ, et.lme, et.se)
push!(rows, (run="truth", β0=NaN, γ0=NaN, ψ0=NaN, β=truth.β, γ=truth.γ, ψ=truth.ψ,
             loglik=et.lme, se=et.se, sec=0.0))
for (s, θ0) ∈ enumerate(starts)
    t = @elapsed fit = mif(tr; Np=NP, Nmif=NMIF, perturbations=kernel, cooling,
                           params=θ0, rng=Xoshiro(100+s))
    θ = fit.params
    e = evaluate(θ)
    @printf("start %d  (%.2f,%.2f,%.3f) → β=%.3f γ=%.3f ψ=%.4f  logLik=%.2f ± %.2f  [%.0fs]\n",
            s, θ0.β, θ0.γ, θ0.ψ, θ.β, θ.γ, θ.ψ, e.lme, e.se, t)
    push!(rows, (run="mif$s", β0=θ0.β, γ0=θ0.γ, ψ0=θ0.ψ, β=θ.β, γ=θ.γ, ψ=θ.ψ,
                 loglik=e.lme, se=e.se, sec=t))
    write_csv(joinpath(RESULTS, "e19_mif_trace$s.csv"),
              [(iteration=i-1, β=p.β, γ=p.γ, ψ=p.ψ, loglik=(i > 1 ? fit.logliks[i-1] : NaN))
               for (i, p) ∈ enumerate(fit.trace)])
end
write_csv(joinpath(RESULTS, "e19_mif.csv"), rows)
end

## ---------------- Part B: profile likelihood for β with a Monte Carlo-adjusted CI
## (MCAP; Ionides, Breto, Park, Smith & King 2017): at each β, IF2 over (γ, ψ)
## from two starts, keep the better evaluated endpoint, then fit a weighted
## quadratic whose coefficient covariance gives the Monte Carlo part of the
## CI cutoff δ = χ²₁(0.95)(a·se_mc² + 1/2).
mcap(θ, ll; level=0.95, span=0.75) = begin
    imax = argmax(ll)
    d = abs.(θ .- θ[imax])
    dmax = sort(d)[max(3, ceil(Int, span*length(d)))]
    w = [di ≤ dmax ? (1 - (di/(dmax*1.0001))^3)^3 : 0.0 for di ∈ d]
    X = hcat(ones(length(θ)), θ, -θ.^2)
    XtW = X' .* w'
    A = XtW*X
    coef = A \ (XtW*ll)
    r = ll .- X*coef
    σ² = sum(w .* r.^2)/(count(>(0), w) - 3)
    V = σ² .* inv(A)
    b, a = coef[2], coef[3]
    vb, va, cab = V[2,2], V[3,3], V[2,3]
    se_mc² = (vb - 2*(b/a)*cab + (b/a)^2*va)/(4a^2)
    δ = 3.841458820694124*(a*se_mc² + 0.5)
    mle = b/(2a)
    half = sqrt(δ/a)
    (; mle, lo=mle-half, hi=mle+half, δ, se_stat=sqrt(1/(2a)), se_mc=sqrt(max(se_mc²,0.0)), a, b, c=coef[1])
end

if occursin('B', PARTS)
    kernel2(scale, lag; γ, ψ, _...) = (γ = γ*exp(scale*RW*randn()), ψ = ψ*exp(scale*RW*randn()))
    Random.seed!(1920)
    prows = NamedTuple[]
    for β ∈ 2.0:0.5:7.0, rep ∈ 1:2
        θ0 = merge(truth, (β=β, γ=0.5*4^rand(), ψ=0.02*5^rand()))
        fit = mif(tr; Np=NP, Nmif=NMIF, perturbations=kernel2, cooling, params=θ0,
                  rng=Xoshiro(hash((β,rep))))
        θ = fit.params
        ev = evaluate(θ)
        push!(prows, (β=β, rep=rep, γ=θ.γ, ψ=θ.ψ, loglik=ev.lme, se=ev.se))
        @printf("β=%.1f rep %d → γ=%.3f ψ=%.4f  logLik=%.2f ± %.2f\n", β, rep, θ.γ, θ.ψ, ev.lme, ev.se)
    end
    write_csv(joinpath(RESULTS, "e19_profile_beta.csv"), prows)
    best = [argmax(r -> r.loglik, filter(r -> r.β == β, prows)) for β ∈ unique(r.β for r ∈ prows)]
    m = mcap([r.β for r ∈ best], [r.loglik for r ∈ best])
    @printf("MCAP: β̂ = %.2f, 95%% CI [%.2f, %.2f]; se_stat=%.2f se_mc=%.2f δ=%.2f (truth β=%.1f)\n",
            m.mle, m.lo, m.hi, m.se_stat, m.se_mc, m.δ, truth.β)
    write_csv(joinpath(RESULTS, "e19_profile_mcap.csv"), [m])
end
