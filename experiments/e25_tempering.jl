## E25: exact data tempering for parameter search on seir_trees[2].
## IF2 (FastSEIR mif) from poor starts, either directly on the whole tree or
## in stages: first on a thinned tree (each sample kept with probability p,
## which for χ = 0 is the same model with ψ replaced by pψ) or on the tree
## curtailed at an earlier time, then on larger versions, warm-starting each
## stage from the previous one.  Every endpoint is evaluated on the whole tree.
include("common.jl")

const NP = parse(Int, get(ENV, "E25_NP", "2000"))
const NSTART = parse(Int, get(ENV, "E25_NSTART", "4"))
const PASS_SD = 0.2         # random-walk SD per filtering pass, on the log scale
const RWPOP = 0.1           # population size, perturbed at lag 0 only

g = parse_newick(FastSEIR.seir_trees[2])
T = g.time
full = FastSEIR.Tree(g, default_guide())
cooling(m) = 0.5^(m/50)

## per-event random-walk SD so that a pass over `tr` perturbs by about PASS_SD
kernel(tr) = begin
    rw = PASS_SD/sqrt(length(tr.time))
    (scale, lag; β, σ, γ, ω, ψ, pop, _...) -> (
        β = β*exp(scale*rw*randn()), σ = σ*exp(scale*rw*randn()), γ = γ*exp(scale*rw*randn()),
        ω = ω*exp(scale*rw*randn()), ψ = ψ*exp(scale*rw*randn()),
        pop = lag == 0 ? max(20.0, round(pop*exp(scale*RWPOP*randn()))) : pop,
    )
end

## a stage: (label, genealogy, sampling fraction relative to the whole tree, iterations)
strategies = [
    ("direct", [("whole", g, 1.0, 60)]),
    ("thinned", [("p=0.1", :thin, 0.1, 40), ("p=0.3", :thin, 0.3, 20), ("whole", g, 1.0, 20)]),
    ("curtailed", [("T/4", T/4, 1.0, 40), ("T/2", T/2, 1.0, 20), ("whole", g, 1.0, 20)]),
]

run_strategy(stages, θ0, seed) = begin
    rng = Xoshiro(seed)
    θ = θ0
    frac = 1.0
    log = String[]
    for (label, what, p, Nmif) ∈ stages
        h = what === :thin ? thin(g, p; rng) : what isa Real ? curtail(g, what) : what
        θ = merge(θ, (ψ = θ.ψ*p/frac,))        # rescale ψ to this stage's sampling fraction
        frac = p
        tr = FastSEIR.Tree(h, default_guide())
        Random.seed!(seed)
        fit = mif(tr; Np=NP, Nmif, perturbations=kernel(tr), cooling, θ..., rng)
        θ = merge(θ, coef(fit))
        push!(log, @sprintf("%s(%d samples): logLik %.1f", label, h.nsample, logLik(fit)))
    end
    θ, join(log, "; ")
end

evaluate(θ) = begin
    θ = merge(θ, (pop = round(Int, θ.pop),))
    ll = [logLik(pfilter(full; Np=4000, θ...)) for _ ∈ 1:8]
    (; mean = mean(ll), sd = std(ll), lme = logmeanexp(ll))
end

Random.seed!(2525)
starts = [("defaults", DEFAULTS);
          [("random$s", merge(DEFAULTS, (β=2*4^rand(), σ=0.5*4^rand(), γ=0.5*4^rand(),
                                          ω=0.5*4^rand(), ψ=0.01*5^rand(), pop=round(Int, 80*3^rand()))))
           for s ∈ 1:NSTART-1]]
rows = NamedTuple[]
for (sname, θ0) ∈ starts, (strategy, stages) ∈ strategies
    local θ, notes
    t = @elapsed try
        θ, notes = run_strategy(stages, merge(θ0, (pop = Float64(θ0.pop),)), hash((sname, strategy)) % 10^6)
    catch err
        θ, notes = nothing, sprint(showerror, err)
    end
    if isnothing(θ)
        @printf("%-9s %-10s FAILED: %s [%.0fs]\n", sname, strategy, notes, t)
        push!(rows, (start=sname, strategy=strategy, β=NaN, σ=NaN, γ=NaN, ω=NaN, ψ=NaN, pop=NaN,
                     mean=NaN, sd=NaN, lme=NaN, sec=t, notes=notes))
    else
        ev = evaluate(θ)
        @printf("%-9s %-10s β=%.2f σ=%.2f γ=%.2f ω=%.2f ψ=%.4f pop=%.0f  mean=%.1f sd=%.2f  [%.0fs] %s\n",
                sname, strategy, θ.β, θ.σ, θ.γ, θ.ω, θ.ψ, θ.pop, ev.mean, ev.sd, t, notes)
        push!(rows, (start=sname, strategy=strategy, β=θ.β, σ=θ.σ, γ=θ.γ, ω=θ.ω, ψ=θ.ψ, pop=θ.pop,
                     mean=ev.mean, sd=ev.sd, lme=ev.lme, sec=t, notes=notes))
    end
    write_csv(joinpath(RESULTS, "e25_tempering.csv"), rows)
end
