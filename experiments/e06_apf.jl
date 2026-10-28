## E06: auxiliary (look-ahead) particle filter in FastSEIR.
## Part A: exactness against ExactSEIR on tiny cases.
## Part B: efficiency on seir_trees[1] at Np = 1000 (single thread).
include("common.jl")
using PhyloPOMP: FastSEIR

const RA = parse(Int, get(ENV, "E06_RA", "200"))
const NPA = parse(Int, get(ENV, "E06_NPA", "500"))
const RB = parse(Int, get(ENV, "E06_RB", "30"))
const NPB = parse(Int, get(ENV, "E06_NPB", "1000"))
mkguide(g) = guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!)

configs = [
    ("baseline local/post/guided", FastSEIR.Options(threaded=false)),
    ("apf lineage",                FastSEIR.Options(threaded=false, lookahead=:lineage)),
    ("apf pop h=1",                FastSEIR.Options(threaded=false, lookahead=:pop, horizon=1)),
    ("apf pop h=3",                FastSEIR.Options(threaded=false, lookahead=:pop, horizon=3)),
    ("apf both h=1",               FastSEIR.Options(threaded=false, lookahead=:both, horizon=1)),
    ("apf both h=3 τ=0.5",         FastSEIR.Options(threaded=false, lookahead=:both, horizon=3, temper=0.5)),
]

## ---------------- Part A
base = (β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.4, χ=0.0)
cases = [
    (name="H1_single_infective", g=parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8),
     par=merge(base, (β=4.0, σ=0.7, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6))),
    (name="H1_branch", g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);", time=1.6),
     par=merge(base, (β=4.0, σ=0.8, pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7))),
]
let k = 0, tries = 0
    Random.seed!(20261028)
    while k < 4 && tries < 2000
        tries += 1
        par = merge(base, (pop=10, S0=0.7, E0=0.1, I0=0.1, R0=0.1))
        g = SimSEIR.simulate(; par..., time=3.0).genealogy
        (3 ≤ g.nsample ≤ 6) || continue
        k += 1
        push!(cases, (name="sim$(k)_n$(g.nsample)", g=g, par=par))
    end
end
Random.seed!(606)
rowsA = NamedTuple[]
for c ∈ cases
    exact = ExactSEIR.loglik(c.g; c.par...)
    tr = FastSEIR.Tree(c.g, mkguide(c.g))
    @printf("\n== %s: exact = %.4f\n", c.name, exact)
    for (label, opt) ∈ configs
        ll = FastSEIR.logliks(tr, c.par; Np=NPA, n=RA, opt)
        s = summarize(ll)
        z = (s.lme-exact)/s.se
        @printf("  %-28s lme=% .4f ± %.4f sd=%.3f z=% .2f\n", label, s.lme, s.se, s.sd, z)
        push!(rowsA, (case=c.name, config=label, exact=exact, lme=s.lme, se=s.se, sd=s.sd, z=z))
    end
end
write_csv(joinpath(RESULTS, "e06_apf_exactness.csv"), rowsA)

## ---------------- Part B
g = parse_newick(GuidedSEIR.seir_trees[1], time=50.0)
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
tr = FastSEIR.Tree(g, mkguide(g))
rowsB = NamedTuple[]
println("\n== seir_trees[1], Np=$(NPB), R=$(RB)")
for (label, opt) ∈ configs
    FastSEIR.pfilter(tr, par; Np=50, opt)
    ll = Float64[]; times = Float64[]; zs = Float64[]
    for r ∈ 1:RB
        st = @timed FastSEIR.pfilter(tr, par; Np=NPB, opt, diagnostics=true)
        push!(ll, st.value.logLik); push!(times, st.time)
        push!(zs, mean(filter(!isnan, st.value.zerofrac)))
    end
    s = summarize(ll); tm = mean(times)
    @printf("  %-28s lme=% .2f ± %.2f sd=%.2f t=%.3fs zero=%.3f t@sd1=%.3fs\n",
            label, s.lme, s.se, s.sd, tm, mean(zs), tm*s.sd^2)
    push!(rowsB, (config=label, lme=s.lme, se=s.se, sd=s.sd, sec_per_run=tm, zero=mean(zs), sec_to_sd1=tm*s.sd^2))
end
write_csv(joinpath(RESULTS, "e06_apf_tree1.csv"), rowsB)
