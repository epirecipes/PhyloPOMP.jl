## E12: does the filter guide matter, and does a guide matched to the model's
## backward-in-time lineage dynamics beat the ad hoc default?
include("common.jl")

const R = parse(Int, get(ENV, "E12_R", "40"))
const NP = parse(Int, get(ENV, "E12_NP", "1000"))
Random.seed!(1212)

g1 = parse_newick(FastSEIR.seir_trees[1], time=50.0)
par = DEFAULTS
@printf("equilibrium: %s\n", equilibrium(; par...))

guides = Any[
    ("none", nothing),
    ("default (πE=0.09, c=1)", default_guide()),
    ("matched (πE=0.5, c=2)", matched_guide(par)),
]
for πE ∈ (0.1, 0.3, 0.5, 0.7), c ∈ (0.5, 1.0, 2.0, 4.0, 8.0)
    push!(guides, ("grid πE=$πE c=$c", fsmarkov(Expos=>πE, Infec=>1-πE, (Expos,Infec)=>c)))
end

rows = NamedTuple[]
println("\n== seir_trees[1], Np=$NP, R=$R")
for (label, m) ∈ guides
    tr = isnothing(m) ? FastSEIR.Tree(g1) : FastSEIR.Tree(g1, m)
    ll, t = logliks(tr, R; Np=NP, par...)
    s = summarize(ll)
    zf = mean(mean(pfilter(tr; Np=NP, par...).zero_fraction) for _ ∈ 1:5)
    @printf("  %-26s lme=% .2f ± %.2f  sd=%.3f  zero=%.3f  t=%.3fs\n", label, s.lme, s.se, s.sd, zf, t)
    push!(rows, (tree="tree1", guide=label, Np=NP, R=R, lme=s.lme, se=s.se, sd=s.sd, zero=zf, sec=t))
end
write_csv(joinpath(RESULTS, "e12_guide_tree1.csv"), rows)
