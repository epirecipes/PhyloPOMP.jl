## E10: scaling with genealogy size and population size on simulated trees.
include("common.jl")
using PhyloPOMP: FastSEIR
using Base.Threads: nthreads

const R = parse(Int, get(ENV, "E10_R", "8"))
mkguide(g) = guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!)
base = (β=4.0, σ=1.0, γ=1.0, ω=1.0, χ=0.0, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
scenarios = [
    (name="N100_T25",  par=merge(base, (pop=100,  ψ=0.02)),  T=25.0),
    (name="N100_T50",  par=merge(base, (pop=100,  ψ=0.02)),  T=50.0),
    (name="N100_T100", par=merge(base, (pop=100,  ψ=0.02)),  T=100.0),
    (name="N100_T200", par=merge(base, (pop=100,  ψ=0.02)),  T=200.0),
    (name="N1000_T50", par=merge(base, (pop=1000, ψ=0.002)), T=50.0),
    (name="N1000_T100",par=merge(base, (pop=1000, ψ=0.002)), T=100.0),
]
rows = NamedTuple[]
for (si, sc) ∈ enumerate(scenarios)
    rng = Xoshiro(1000 + si)
    sim = SimSEIR.simulate(; sc.par..., time=sc.T, rng)
    while sim.nsample < 8
        sim = SimSEIR.simulate(; sc.par..., time=sc.T, rng)
    end
    g = sim.genealogy
    tr = FastSEIR.Tree(g, mkguide(g))
    @printf("\n== %s: samples=%d nodes=%d popevents=%d\n", sc.name, g.nsample, length(g), sim.nevents)
    methods = [
        ("pomp/guided", 1000, () -> logLik(pfilter(make_pomp(:guided, g; sc.par...); Np=1000))),
        ("fast/naive/pre", 1000, () -> FastSEIR.pfilter(tr, sc.par; Np=1000,
             opt=FastSEIR.Options(proposal=:naive, guided=false, order=:pre)).logLik),
        ("fast/best", 1000, () -> FastSEIR.pfilter(tr, sc.par; Np=1000).logLik),
        ("fast/best", 4000, () -> FastSEIR.pfilter(tr, sc.par; Np=4000).logLik),
    ]
    for (label, Np, f) ∈ methods
        f()                           # compile / warm up
        ll = Float64[]; tt = Float64[]
        for r ∈ 1:R
            st = @timed f(); push!(ll, st.value); push!(tt, st.time)
        end
        s = summarize(ll); tm = mean(tt)
        @printf("  %-15s Np=%5d lme=% .2f ± %.2f sd=%.2f finite=%d/%d t=%.3fs t@sd1=%.3fs\n",
                label, Np, s.lme, s.se, s.sd, s.nfinite, R, tm, tm*s.sd^2)
        push!(rows, (scenario=sc.name, nsample=Int(g.nsample), nodes=length(g), pop=sc.par.pop,
                     T=sc.T, method=label, Np=Np, threads=(label == "pomp/guided" ? 1 : nthreads()),
                     lme=s.lme, se=s.se, sd=s.sd, nfinite=s.nfinite, R=R, sec_per_run=tm,
                     sec_to_sd1=tm*s.sd^2))
    end
end
write_csv(joinpath(RESULTS, "e10_scaling.csv"), rows)
