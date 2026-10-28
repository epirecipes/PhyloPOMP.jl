## E08: wall-clock scaling of FastSEIR with threads (reproducible chunked RNG streams).
include("common.jl")
using PhyloPOMP: FastSEIR
using Base.Threads: nthreads
g = parse_newick(GuidedSEIR.seir_trees[parse(Int, get(ENV,"E08_TREE","1"))], time=parse(Float64, get(ENV,"E08_TIME","50")))
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
tr = FastSEIR.Tree(g, guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!))
rows = NamedTuple[]
for Np ∈ (1000, 10000), threaded ∈ (false, true)
    opt = FastSEIR.Options(threaded=threaded, nchunks=64)
    FastSEIR.pfilter(tr, par; Np=200, opt)
    ts = [@elapsed(FastSEIR.pfilter(tr, par; Np, opt)) for _ ∈ 1:5]
    ## reproducibility: same master seed ⇒ same estimate regardless of threading
    l1 = FastSEIR.pfilter(tr, par; Np, opt, rng=Xoshiro(1)).logLik
    l2 = FastSEIR.pfilter(tr, par; Np, opt=FastSEIR.Options(threaded=!threaded, nchunks=64), rng=Xoshiro(1)).logLik
    @printf("threads=%d Np=%5d threaded=%-5s median time=%.3fs  reproducible across threading: %s (%.6f vs %.6f)\n",
            nthreads(), Np, threaded, median(ts), l1 == l2, l1, l2)
    push!(rows, (nthreads=nthreads(), Np=Np, threaded=threaded, median_sec=median(ts), reproducible=(l1 == l2)))
end
write_csv(joinpath(RESULTS, "e08_threads_t$(nthreads()).csv"), rows)
