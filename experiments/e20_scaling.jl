## E20: particle-number scaling on seir_trees[2] (864 samples): does the SD of
## the log-likelihood estimate fall as 1/√Np, and what does SD ≈ 1 cost?
include("common.jl")
include("weighted_filter.jl")
using .WFilter: pfilter_w

const R = parse(Int, get(ENV, "E20_R", "8"))
const NPS = parse.(Int, split(get(ENV, "E20_NPS", "16000,64000"), ","))
Random.seed!(2020)

g = parse_newick(FastSEIR.seir_trees[2])
methods = [
    ("baseline", FastSEIR.Tree(g, default_guide()), (tr; kw...) -> pfilter(tr; kw...)),
    ("twist-notilt+matched", FastSEIR.Tree(g, matched_guide(DEFAULTS)),
     (tr; kw...) -> pfilter_w(tr; twist=true, tilt=false, kw...)),
]
rows = NamedTuple[]
for Np ∈ NPS, (label, tr, f) ∈ methods
    f(tr; Np=50, DEFAULTS...)
    res = [(@timed f(tr; Np, DEFAULTS...)) for _ ∈ 1:R]
    ll = [logLik(r.value) for r ∈ res]
    t = mean(r.time for r ∈ res)
    s = summarize(ll)
    @printf("Np=%6d  %-21s mean=% .2f ± %.2f  sd=%.2f  lme=%.2f  max=%.2f  t=%.1fs\n",
            Np, label, mean(ll), s.sd/sqrt(R), s.sd, s.lme, maximum(ll), t)
    push!(rows, (method=label, Np=Np, R=R, mean=mean(ll), sd=s.sd, lme=s.lme, max=maximum(ll), sec=t))
    write_csv(joinpath(RESULTS, "e20_scaling_tree2.csv"), rows)
end
