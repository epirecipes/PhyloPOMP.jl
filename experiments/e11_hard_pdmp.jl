## E11: time-varying "hard" proposals (support-correct): rates frozen at the last
## jump (exact piecewise-constant hazards) vs exact time variation via
## PiecewiseDeterministicMarkovProcesses.jl, with and without a forward-ODE
## population twist.  Part A: exactness vs ExactSEIR.  Part B: tree-1 efficiency.
include("common.jl")
include("pdmp_proposals.jl")
using PhyloPOMP: FastSEIR
using .PDMPProposals: HardCfg, options

const RA = parse(Int, get(ENV, "E11_RA", "150"))
const NPA = parse(Int, get(ENV, "E11_NPA", "200"))
const RB = parse(Int, get(ENV, "E11_RB", "30"))
const NPB = parse(Int, get(ENV, "E11_NPB", "500"))
mkguide(g) = guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!)

configs = [
    ("baseline local/post/guided", FastSEIR.Options(threaded=false)),
    ("hard frozen",                options(:frozen; cfg=HardCfg(guided=true))),
    ("hard pdmp",                  options(:pdmp;   cfg=HardCfg(guided=true))),
    ("hard frozen + twist",        options(:frozen; cfg=HardCfg(guided=true, twist=true, horizon=2))),
    ("hard pdmp + twist",          options(:pdmp;   cfg=HardCfg(guided=true, twist=true, horizon=2))),
]

base = (β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.4, χ=0.0)
cases = [
    (name="H1_single_infective", g=parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8),
     par=merge(base, (β=4.0, σ=0.7, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6))),
    (name="H1_branch", g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);", time=1.6),
     par=merge(base, (β=4.0, σ=0.8, pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7))),
]
let k = 0, tries = 0
    Random.seed!(20261028)
    while k < 3 && tries < 2000
        tries += 1
        par = merge(base, (pop=10, S0=0.7, E0=0.1, I0=0.1, R0=0.1))
        g = SimSEIR.simulate(; par..., time=3.0).genealogy
        (3 ≤ g.nsample ≤ 6) || continue
        k += 1
        push!(cases, (name="sim$(k)_n$(g.nsample)", g=g, par=par))
    end
end

Random.seed!(1111)
rowsA = NamedTuple[]
for c ∈ cases
    exact = ExactSEIR.loglik(c.g; c.par...)
    tr = FastSEIR.Tree(c.g, mkguide(c.g))
    @printf("\n== %s: exact = %.4f\n", c.name, exact)
    for (label, opt) ∈ configs
        t = @elapsed ll = FastSEIR.logliks(tr, c.par; Np=NPA, n=RA, opt)
        s = summarize(ll)
        z = (s.lme-exact)/s.se
        @printf("  %-28s lme=% .4f ± %.4f sd=%.3f z=% .2f (%.1fs)\n", label, s.lme, s.se, s.sd, z, t)
        push!(rowsA, (case=c.name, config=label, exact=exact, lme=s.lme, se=s.se, sd=s.sd, z=z, seconds=t))
    end
end
write_csv(joinpath(RESULTS, "e11_hard_exactness.csv"), rowsA)

g = parse_newick(GuidedSEIR.seir_trees[1], time=50.0)
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
tr = FastSEIR.Tree(g, mkguide(g))
rowsB = NamedTuple[]
println("\n== seir_trees[1], Np=$(NPB), R=$(RB) (single thread)")
for (label, opt) ∈ configs
    FastSEIR.pfilter(tr, par; Np=20, opt)
    ll = Float64[]; tt = Float64[]; zs = Float64[]
    for r ∈ 1:RB
        st = @timed FastSEIR.pfilter(tr, par; Np=NPB, opt, diagnostics=true)
        push!(ll, st.value.logLik); push!(tt, st.time)
        push!(zs, mean(filter(!isnan, st.value.zerofrac)))
    end
    s = summarize(ll); tm = mean(tt)
    @printf("  %-28s lme=% .2f ± %.2f sd=%.2f zero=%.3f t=%.2fs t@sd1=%.2fs\n",
            label, s.lme, s.se, s.sd, mean(zs), tm, tm*s.sd^2)
    push!(rowsB, (config=label, Np=NPB, R=RB, lme=s.lme, se=s.se, sd=s.sd, zero=mean(zs),
                  sec_per_run=tm, sec_to_sd1=tm*s.sd^2))
end
write_csv(joinpath(RESULTS, "e11_hard_tree1.csv"), rowsB)
