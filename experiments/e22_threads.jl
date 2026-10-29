## E22: thread scaling of FastSEIR.  Run once per thread count, e.g.
##   for n in 1 2 4 8; do julia -t $n --project=experiments experiments/e22_threads.jl; done
## E22_VERSION labels the code version (the active project's PhyloPOMP) and
## E22_CSV names the output file, to which a row per case is appended.
include("common.jl")

const LABEL = get(ENV, "E22_VERSION", "perf/threading")
const CSV = get(ENV, "E22_CSV", joinpath(RESULTS, "e22_threads.csv"))
const REPS = parse(Int, get(ENV, "E22_REPS", "3"))

## the E21b estimate for seir_trees[2]
fitted = merge(DEFAULTS, (β=2.806, σ=0.599, γ=0.478, ω=5.171, ψ=0.0271, pop=173))
cases = [
    ("tree1", parse_newick(FastSEIR.seir_trees[1], time=50.0), DEFAULTS, 10000),
    ("tree2", parse_newick(FastSEIR.seir_trees[2]), fitted, 4000),
]

isfile(CSV) || open(io -> println(io, "version,threads,tree,Np,sec,logLik,nevents"), CSV, "w")
for (name, g, par, Np) ∈ cases
    tr = FastSEIR.Tree(g, default_guide())
    pfilter(tr; Np=100, par...)
    res = [(@timed pfilter(tr; Np, par..., rng=Xoshiro(r))) for r ∈ 1:REPS]
    sec = median(r.time for r ∈ res)
    ## the estimate for a given rng must not depend on the number of threads
    ll = logLik(res[1].value)
    @printf("%-16s threads=%d %s Np=%d: %.2fs  logLik(rng 1)=%.6f\n", LABEL, Threads.nthreads(), name, Np, sec, ll)
    open(CSV, "a") do io
        println(io, join(("\"$LABEL\"", Threads.nthreads(), "\"$name\"", Np, sec, repr(ll), res[1].value.nevents), ","))
    end
end
