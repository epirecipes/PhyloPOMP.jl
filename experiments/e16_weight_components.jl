## E16: where does the weight variance come from?  Per-interval decomposition
## of the log weights (experiments/weight_components.jl) on seir_trees[1] and [2].
include("common.jl")
include("weight_components.jl")
using .WeightDiag: diagnose

const NP = parse(Int, get(ENV, "E16_NP", "2000"))
comps = (:phi, :is, :decay, :event, :total)

report(label, rows) = begin
    println("\n== $label  ($(length(rows)) intervals, Np=$NP)")
    for sel ∈ ("all", "Sample", "Node")
        rs = sel == "all" ? rows : filter(r -> r.closing == sel, rows)
        isempty(rs) && continue
        @printf("  %-6s n=%4d  fail: parentE=%.3f S0=%.4f root=%.4f\n", sel, length(rs),
                mean(r.fail_parentE for r ∈ rs), mean(r.fail_S0 for r ∈ rs), mean(r.fail_root for r ∈ rs))
        @printf("         mean var:  "); foreach(c -> @printf("%s=%.3f  ", c, mean(getproperty(r, Symbol("var_",c)) for r ∈ rs)), comps); println()
        @printf("         mean ESS:  "); foreach(c -> @printf("%s=%.3f  ", c, mean(filter(isfinite, [getproperty(r, Symbol("ess_",c)) for r ∈ rs]))), comps); println()
    end
end

allrows = NamedTuple[]
for (tree, time) ∈ ((1, 50.0), (2, nothing))
    g = isnothing(time) ? parse_newick(FastSEIR.seir_trees[tree]) : parse_newick(FastSEIR.seir_trees[tree]; time)
    for (gname, gd) ∈ (("default", default_guide()), ("matched", matched_guide(DEFAULTS)), ("none", nothing)),
        weighted ∈ (false, true)
        tr = isnothing(gd) ? FastSEIR.Tree(g) : FastSEIR.Tree(g, gd)
        rows = diagnose(tr; Np=NP, DEFAULTS..., weighted, seed=UInt64(16))
        report("seir_trees[$tree], guide=$gname, rates=$(weighted ? "weighted" : "prior")", rows)
        append!(allrows, [merge((tree=tree, guide=gname, rates=weighted ? "weighted" : "prior"), r) for r ∈ rows])
    end
end
write_csv(joinpath(RESULTS, "e16_weight_components.csv"), allrows)
