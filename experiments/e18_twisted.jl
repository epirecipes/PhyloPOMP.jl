## E18: twisted (Doob-style) proposals (experiments/weighted_filter.jl).
## Part A: exactness with large Np against ExactSEIR (also covers the E15 APF).
## Part B: efficiency on seir_trees[1] and [2]; judge by the mean log likelihood
## (all estimators are unbiased on the natural scale, so a higher mean of the
## log estimate means a smaller downward bias) as well as its SD.
include("common.jl")
include("weighted_filter.jl")
include("apf_filter.jl")
using .WFilter: pfilter_w
using .APF: pfilter_apf

const RA = parse(Int, get(ENV, "E18_RA", "40"))
const R1 = parse(Int, get(ENV, "E18_R1", "40"))
const R2 = parse(Int, get(ENV, "E18_R2", "16"))
const NP2 = parse(Int, get(ENV, "E18_NP2", "4000"))
const PARTS = get(ENV, "E18_PARTS", "AB")
Random.seed!(1818)

methods = [
    ("baseline", default_guide(), (tr; kw...) -> pfilter(tr; kw...)),
    ("weighted", default_guide(), (tr; kw...) -> pfilter_w(tr; kw...)),
    ("twist", default_guide(), (tr; kw...) -> pfilter_w(tr; twist=true, kw...)),
    ("twist-notilt", default_guide(), (tr; kw...) -> pfilter_w(tr; twist=true, tilt=false, kw...)),
    ("twist-notilt+matched", matched_guide(DEFAULTS), (tr; kw...) -> pfilter_w(tr; twist=true, tilt=false, kw...)),
]

if occursin('A', PARTS)
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
        @printf("== %-20s exact=%.4f\n", c.name, exact)
        for (label, gd, f) ∈ [methods; ("apf τ=1", default_guide(), (tr; kw...) -> pfilter_apf(tr; temper=1.0, kw...))]
            tr = FastSEIR.Tree(c.g, gd)
            ll = [logLik(f(tr; Np=20000, c.par...)) for _ ∈ 1:RA]
            d, se = mean(ll)-exact, std(ll)/sqrt(RA)
            @printf("   %-21s mean-exact=% .4f ± %.4f  z=% .2f  sd=%.4f\n", label, d, se, d/se, std(ll))
            push!(rowsA, (case=c.name, method=label, exact=exact, bias=d, se=se, z=d/se, sd=std(ll)))
        end
    end
    write_csv(joinpath(RESULTS, "e18_twisted_exactness.csv"), rowsA)
end

if occursin('B', PARTS)
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
            zf = mean(mean(r.value.zero_fraction) for r ∈ res)
            s = summarize(ll)
            @printf("  %-21s mean=% .2f ± %.2f  sd=%.2f  lme=%.2f  max=%.2f  ESS/Np=%.3f  zero=%.3f  t=%.2fs\n",
                    label, mean(ll), s.sd/sqrt(R), s.sd, s.lme, maximum(ll), ess, zf, t)
            push!(rowsB, (tree=tree, method=label, Np=Np, R=R, mean=mean(ll), sd=s.sd, lme=s.lme,
                          max=maximum(ll), ess_frac=ess, zero_frac=zf, sec=t))
        end
    end
    write_csv(joinpath(RESULTS, "e18_twisted_efficiency.csv"), rowsB)
end
