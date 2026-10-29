## E24: inference from FastSEIR's sufficient statistics.
## Part A: Monte Carlo EM (FastSEIR.em) and an accelerated variant from the
##         eight random starts of E19 on its 92-sample genealogy, against IF2.
## Part B: precision of the Fisher-identity score, and the fraction of
##         missing information that sets EM's rate of convergence.
## Part C: the score on seir_trees[2] at the defaults and at the E21b estimate.
include("common.jl")

const PARTS = get(ENV, "E24_PARTS", "ABC")
const NEM = parse(Int, get(ENV, "E24_NEM", "30"))
const NP = parse(Int, get(ENV, "E24_NP", "1000"))

truth = merge(DEFAULTS, (ψ=0.05,))
g = let rng = Xoshiro(1919), sim = nothing
    while true
        sim = SimSEIR.simulate(; truth..., time=60.0, rng)
        80 ≤ sim.genealogy.nsample ≤ 160 && break
    end
    sim.genealogy
end
tr = FastSEIR.Tree(g, default_guide())
println("E19 genealogy: $(g.nsample) samples, $(length(tr.time)) events")
evaluate(θ; R=10, Np=2000) = summarize([logLik(pfilter(tr; Np, θ...)) for _ ∈ 1:R])
cmle(s) = (β = s.transmissions/s.SI, γ = s.recoveries/s.I, ψ = s.samples/s.I)

## EM whose step for each rate (log scale) is multiplied by a factor that grows
## by half while successive steps agree in sign (up to 50) and halves otherwise
accelerated_em(θ, est; Np, Nem, rng) = begin
    mult = Dict(x => 1.0 for x ∈ est); prev = Dict(x => 0.0 for x ∈ est)
    for _ ∈ 1:Nem
        mle = cmle(pfilter(tr; Np, θ..., rng).suffstats)
        new = map(est) do x
            d = log(mle[x]/θ[x])
            mult[x] = sign(d) == sign(prev[x]) ? min(1.5mult[x], 50.0) : max(mult[x]/2, 1.0)
            prev[x] = d
            θ[x]*exp(mult[x]*d)
        end
        θ = merge(θ, NamedTuple{est}(Tuple(new)))
    end
    θ
end

## ---------------- Part A
if occursin('A', PARTS)
    Random.seed!(19)            # the starts of E19
    starts = [merge(truth, (β=2*4^rand(), γ=0.5*4^rand(), ψ=0.02*5^rand())) for _ ∈ 1:8]
    rows = NamedTuple[]
    for (s, θ0) ∈ enumerate(starts), method ∈ ("MCEM", "accelerated EM")
        t = @elapsed θ = method == "MCEM" ?
            coef(FastSEIR.em(tr; Np=NP, Nem=NEM, estimate=(:β,:γ,:ψ), θ0..., rng=Xoshiro(2400+s))) :
            accelerated_em(θ0, (:β,:γ,:ψ); Np=NP, Nem=NEM, rng=Xoshiro(2400+s))
        ev = evaluate(θ)
        @printf("start %d (%.2f,%.2f,%.3f) %-14s → β=%.3f γ=%.3f ψ=%.4f  logLik=%.2f ± %.2f  [%.1fs]\n",
                s, θ0.β, θ0.γ, θ0.ψ, method, θ.β, θ.γ, θ.ψ, ev.lme, ev.se, t)
        push!(rows, (method=method, start=s, β0=θ0.β, γ0=θ0.γ, ψ0=θ0.ψ, β=θ.β, γ=θ.γ, ψ=θ.ψ,
                     loglik=ev.lme, se=ev.se, sec=t))
        write_csv(joinpath(RESULTS, "e24_em_e19tree.csv"), rows)
    end
    lines = readlines(joinpath(RESULTS, "e19_mif.csv"))
    col = findfirst(==("loglik"), split(lines[1], ","))
    if2 = [parse(Float64, split(l, ",")[col]) for l ∈ lines[2:end] if startswith(l, "\"mif")]
    println("IF2 (E19, 60 iterations): endpoint logLik ", join([@sprintf("%.2f", x) for x ∈ if2], ", "))
end

## ---------------- Part B
if occursin('B', PARTS)
    mle = merge(truth, (β=3.962, γ=1.048, ψ=0.0690))       # best IF2 endpoint of E19
    rows = NamedTuple[]
    for (label, θ) ∈ (("IF2 estimate", mle), ("truth", truth)), Np ∈ (1000, 4000, 16000)
        pfs = [pfilter(tr; Np, θ..., rng=Xoshiro(r)) for r ∈ 1:12]
        sc = [FastSEIR.score(pf; θ...) for pf ∈ pfs]
        st = [pf.suffstats for pf ∈ pfs]
        f(x) = (mean(getindex.(sc, x)), std(getindex.(sc, x)))
        @printf("%-12s Np=%5d  score β %.1f ± %.1f  γ %.1f ± %.1f  ψ %.1f ± %.1f | E[transmissions]=%.0f E[recoveries]=%.0f E[samples]=%.0f\n",
                label, Np, f(:β)..., f(:γ)..., f(:ψ)...,
                mean(getindex.(st, :transmissions)), mean(getindex.(st, :recoveries)), mean(getindex.(st, :samples)))
        push!(rows, (params=label, Np=Np, score_β=f(:β)[1], sd_β=f(:β)[2], score_γ=f(:γ)[1], sd_γ=f(:γ)[2],
                     score_ψ=f(:ψ)[1], sd_ψ=f(:ψ)[2], transmissions=mean(getindex.(st, :transmissions)),
                     recoveries=mean(getindex.(st, :recoveries)), samples=mean(getindex.(st, :samples))))
    end
    ## complete-data information about log β is E[transmissions]; the observed
    ## information is (β̂/se)² from the MCAP profile of E19 (β̂ = 4.05, se = 0.92)
    info_obs = (4.05/0.92)^2
    info_com = rows[2].transmissions
    @printf("log β: observed information %.1f, complete-data information %.0f → EM rate %.3f\n",
            info_obs, info_com, 1 - info_obs/info_com)
    write_csv(joinpath(RESULTS, "e24_score.csv"), rows)
end

## ---------------- Part C
if occursin('C', PARTS)
    g2 = parse_newick(FastSEIR.seir_trees[2])
    tr2 = FastSEIR.Tree(g2, default_guide())
    fitted = merge(DEFAULTS, (β=2.806, σ=0.599, γ=0.478, ω=5.171, ψ=0.0271, pop=173))
    rows = NamedTuple[]
    for (label, θ) ∈ (("defaults", DEFAULTS), ("E21b estimate", fitted))
        sc = [FastSEIR.score(pfilter(tr2; Np=4000, θ..., rng=Xoshiro(r)); θ...) for r ∈ 1:4]
        m = map(x -> (mean(getindex.(sc, x)), std(getindex.(sc, x))), (β=:β, σ=:σ, γ=:γ, ω=:ω, ψ=:ψ))
        @printf("tree 2 at %-13s score: β %.0f ± %.0f  σ %.0f ± %.0f  γ %.0f ± %.0f  ω %.0f ± %.0f  ψ %.0f ± %.0f\n",
                label, m.β..., m.σ..., m.γ..., m.ω..., m.ψ...)
        push!(rows, (params=label, score_β=m.β[1], sd_β=m.β[2], score_σ=m.σ[1], sd_σ=m.σ[2],
                     score_γ=m.γ[1], sd_γ=m.γ[2], score_ω=m.ω[1], sd_ω=m.ω[2], score_ψ=m.ψ[1], sd_ψ=m.ψ[2]))
    end
    write_csv(joinpath(RESULTS, "e24_score_tree2.csv"), rows)
end
