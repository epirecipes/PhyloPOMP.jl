## E17: weighted-rate proposal (experiments/weighted_filter.jl).
## Part A: exactness against ExactSEIR.  Part B: efficiency on seir_trees[1] and [2].
include("common.jl")
include("weighted_filter.jl")
using .WFilter: pfilter_w

const RA = parse(Int, get(ENV, "E17_RA", "150"))
const R1 = parse(Int, get(ENV, "E17_R1", "40"))
const R2 = parse(Int, get(ENV, "E17_R2", "16"))
const NP2 = parse(Int, get(ENV, "E17_NP2", "4000"))
Random.seed!(1717)

methods = [
    ("baseline", default_guide(), (tr; kw...) -> pfilter(tr; kw...)),
    ("weighted", default_guide(), (tr; kw...) -> pfilter_w(tr; kw...)),
    ("weighted+matched", matched_guide(DEFAULTS), (tr; kw...) -> pfilter_w(tr; kw...)),
]

## ---------------- Part A: exactness
base = (β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.4, χ=0.0)
cases = [
    (name="H1_single_infective", g=parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8),
     par=merge(base, (β=4.0, σ=0.7, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6))),
    (name="H1_branch", g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);", time=1.6),
     par=merge(base, (β=4.0, σ=0.8, pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7))),
]
let k = 0, rng = Xoshiro(20261028)
    while k < 3
        par = merge(base, (pop=10, S0=0.7, E0=0.1, I0=0.1, R0=0.1))
        g = SimSEIR.simulate(; par..., time=3.0, rng).genealogy
        (3 ≤ g.nsample ≤ 6) || continue
        k += 1
        push!(cases, (name="sim$(k)_n$(g.nsample)", g=g, par=par))
    end
end
rowsA = NamedTuple[]
for c ∈ cases
    exact = ExactSEIR.loglik(c.g; c.par...)
    @printf("== %-20s exact=%.4f", c.name, exact)
    for (label, gd, f) ∈ methods[1:2]
        tr = FastSEIR.Tree(c.g, gd)
        s = summarize([logLik(f(tr; Np=300, c.par...)) for _ ∈ 1:RA])
        z = (s.lme-exact)/s.se
        @printf(" | %s z=% .2f sd=%.2f", label, z, s.sd)
        push!(rowsA, (case=c.name, method=label, exact=exact, lme=s.lme, se=s.se, sd=s.sd, z=z))
    end
    println()
end
write_csv(joinpath(RESULTS, "e17_weighted_exactness.csv"), rowsA)

## ---------------- Part B: efficiency
rowsB = NamedTuple[]
for (tree, time, Np, R) ∈ ((1, 50.0, 1000, R1), (2, nothing, NP2, R2))
    g = isnothing(time) ? parse_newick(FastSEIR.seir_trees[tree]) : parse_newick(FastSEIR.seir_trees[tree]; time)
    println("\n== seir_trees[$tree], Np=$Np, R=$R")
    for (label, gd, f) ∈ methods
        tr = FastSEIR.Tree(g, gd)
        f(tr; Np=50, DEFAULTS...)
        res = [(@timed f(tr; Np, DEFAULTS...)) for _ ∈ 1:R]
        ll = [logLik(r.value) for r ∈ res]
        t = mean(r.time for r ∈ res)
        ess = mean(mean(eff_sample_size(r.value))/Np for r ∈ res)
        s = summarize(ll)
        @printf("  %-17s lme=% .2f ± %.2f  mean=%.2f  sd=%.2f  ESS/Np=%.3f  t=%.2fs  t@sd1=%.2fs\n",
                label, s.lme, s.se, mean(ll), s.sd, ess, t, t*s.sd^2)
        push!(rowsB, (tree=tree, method=label, Np=Np, R=R, lme=s.lme, se=s.se, mean=mean(ll),
                      sd=s.sd, ess_frac=ess, sec=t, sec_to_sd1=t*s.sd^2))
    end
end
write_csv(joinpath(RESULTS, "e17_weighted_efficiency.csv"), rowsB)
