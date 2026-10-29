## E23: O(1) color proposals (perf/colour-bookkeeping).
## Part A: exactness against ExactSEIR (this branch's code).
## Part B: time per run, SD of the log likelihood and time to SD 1 (E23_RSCALE
## multiplies the replicates), for the
## code version of the active project (E23_VERSION), appended to E23_CSV, on
## seir_trees[1], seir_trees[2] at its E21b estimate, and a simulated tree
## with many concurrent lineages.
include("common.jl")

const PARTS = get(ENV, "E23_PARTS", "AB")
const LABEL = get(ENV, "E23_VERSION", "perf/colour-bookkeeping")
const CSV = get(ENV, "E23_CSV", joinpath(RESULTS, "e23_colour_speed.csv"))
const RSCALE = parse(Int, get(ENV, "E23_RSCALE", "1"))   # more replicates for precision

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
    Random.seed!(2323)
    for c ∈ cases, (label, kw) ∈ (("guided", (;)), ("unguided", (guided=false,)), ("uniform", (proposal=:uniform,)))
        exact = ExactSEIR.loglik(c.g; c.par...)
        tr = FastSEIR.Tree(c.g, default_guide())
        ll = [logLik(pfilter(tr; Np=20000, c.par..., kw...)) for _ ∈ 1:40]
        d, se = mean(ll)-exact, std(ll)/sqrt(40)
        @printf("== %-20s %-9s mean-exact=% .4f ± %.4f  z=% .2f\n", c.name, label, d, se, d/se)
        push!(rowsA, (case=c.name, proposal=label, exact=exact, bias=d, se=se, z=d/se))
    end
    write_csv(joinpath(RESULTS, "e23_colour_exactness.csv"), rowsA)
end

if occursin('B', PARTS)
    fitted = merge(DEFAULTS, (β=2.806, σ=0.599, γ=0.478, ω=5.171, ψ=0.0271, pop=173))
    dense = merge(DEFAULTS, (pop=1000, ψ=0.05))
    gdense = SimSEIR.simulate(; dense..., time=30.0, rng=Xoshiro(2323)).genealogy
    cases = [
        ("tree1", parse_newick(FastSEIR.seir_trees[1], time=50.0), DEFAULTS, 1000, 40),
        ("tree2", parse_newick(FastSEIR.seir_trees[2]), fitted, 4000, 8),
        ("dense", gdense, dense, 1000, 8),
    ]
    isfile(CSV) || open(io -> println(io, "version,tree,samples,max_lineages,Np,R,sec,mean,sd,sec_to_sd1"), CSV, "w")
    for (name, g, par, Np, R) ∈ cases
        tr = FastSEIR.Tree(g, default_guide())
        nl = maximum(tr.ptr[k+1]-tr.ptr[k] for k ∈ eachindex(tr.time))
        pfilter(tr; Np=50, par...)
        res = [(@timed logLik(pfilter(tr; Np, par...))) for _ ∈ 1:R*RSCALE]
        ll = [r.value for r ∈ res]
        sec = median(r.time for r ∈ res)
        @printf("%-24s %-6s (%d samples, ≤%d lineages) Np=%d: %.2fs/run  mean=%.2f sd=%.2f  t@sd1=%.2fs\n",
                LABEL, name, g.nsample, nl, Np, sec, mean(ll), std(ll), sec*std(ll)^2)
        open(CSV, "a") do io
            println(io, join(("\"$LABEL\"", "\"$name\"", g.nsample, nl, Np, R*RSCALE, sec, mean(ll), std(ll), sec*std(ll)^2), ","))
        end
    end
end
