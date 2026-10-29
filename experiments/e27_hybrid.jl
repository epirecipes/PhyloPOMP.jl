## E27: approximate τ-leap hybrid filter for large populations
## (experiments/hybrid_filter.jl), against the exact FastSEIR filter with the
## same (unguided) proposals.
## Part A: bias as a function of the step h, at N = 100 and N = 1000.
## Part B: cost as a function of N, with ψ scaled to give ~100 samples.
include("common.jl")
include("hybrid_filter.jl")
using .Hybrid: pfilter_hybrid

const PARTS = get(ENV, "E27_PARTS", "AB")
const NP = parse(Int, get(ENV, "E27_NP", "2000"))
const R = parse(Int, get(ENV, "E27_R", "16"))
const HS = parse.(Float64, split(get(ENV, "E27_HS", "0.2,0.1,0.05,0.02,0.01"), ","))

exact(tr; kw...) = pfilter(tr; guided=false, kw...)
runs(f, tr, par; Np, R) = begin
    f(tr; Np=50, par...)
    res = [(@timed f(tr; Np, par...)) for _ ∈ 1:R]
    ll = [logLik(r.value) for r ∈ res]
    e = logmeanexp(ll, se=true)
    (; lme = e.est, se = e.se, mean = mean(ll), sd = std(ll), sec = mean(r.time for r ∈ res),
       nevents = mean(r.value.nevents for r ∈ res)/Np)
end

simulated(N; T=40.0, seed=27) = begin
    par = merge(DEFAULTS, (pop=N, ψ=8/N))
    g = SimSEIR.simulate(; par..., time=T, rng=Xoshiro(seed)).genealogy
    g, par
end

## ---------------- Part A: bias against step size
if occursin('A', PARTS)
    rows = NamedTuple[]
    for (name, g, par) ∈ (("seir_trees[1] (N=100)", parse_newick(FastSEIR.seir_trees[1], time=50.0), DEFAULTS),
                          ("simulated N=1000", simulated(1000)...))
        tr = FastSEIR.Tree(g)
        ref = runs(exact, tr, par; Np=NP, R)
        @printf("== %s: %d samples; exact lme=%.2f ± %.2f sd=%.2f [%.2fs/run, %.0f events/particle]\n",
                name, g.nsample, ref.lme, ref.se, ref.sd, ref.sec, ref.nevents)
        push!(rows, (tree=name, method="exact", h=0.0, lme=ref.lme, se=ref.se, sd=ref.sd,
                     bias=0.0, bias_se=0.0, sec=ref.sec))
        for h ∈ HS
            r = runs((tr; kw...) -> pfilter_hybrid(tr; h, kw...), tr, par; Np=NP, R)
            b, bse = r.lme - ref.lme, sqrt(r.se^2 + ref.se^2)
            @printf("   h=%-5.3g lme=%.2f ± %.2f sd=%.2f  bias=% .2f ± %.2f  [%.2fs/run]\n",
                    h, r.lme, r.se, r.sd, b, bse, r.sec)
            push!(rows, (tree=name, method="hybrid", h=h, lme=r.lme, se=r.se, sd=r.sd,
                         bias=b, bias_se=bse, sec=r.sec))
        end
        write_csv(joinpath(RESULTS, "e27_hybrid_bias.csv"), rows)
    end
end

## ---------------- Part B: cost against population size
if occursin('B', PARTS)
    rows = NamedTuple[]
    hB = 0.01                  # bias ≈ 0.2–0.3 log units in part A
    texact = Ref(0.0)          # seconds per exact run at the previous N
    for N ∈ (10^3, 10^4, 10^5)
        g, par = simulated(N)
        tr = FastSEIR.Tree(g)
        nl = maximum(tr.ptr[k+1]-tr.ptr[k] for k ∈ eachindex(tr.time))
        hyb = runs((tr; kw...) -> pfilter_hybrid(tr; h=hB, kw...), tr, par; Np=1000, R=8)
        ## skip the exact filter when it would take more than ~10 minutes per run
        ex = 10*texact[] > 600 ? nothing : runs(exact, tr, par; Np=1000, R=N ≥ 10^5 ? 2 : 8)
        texact[] = isnothing(ex) ? Inf : ex.sec
        @printf("N=%-7d %3d samples, ≤%3d lineages | hybrid h=%.2f: lme=%.2f sd=%.2f %.2fs/run",
                N, g.nsample, nl, hB, hyb.lme, hyb.sd, hyb.sec)
        isnothing(ex) ? println(" | exact: skipped") :
            @printf(" | exact: lme=%.2f sd=%.2f %.2fs/run (%.0f events/particle)\n", ex.lme, ex.sd, ex.sec, ex.nevents)
        push!(rows, (N=N, samples=g.nsample, max_lineages=nl, h=hB,
                     hybrid_lme=hyb.lme, hybrid_sd=hyb.sd, hybrid_sec=hyb.sec,
                     exact_lme=isnothing(ex) ? NaN : ex.lme, exact_sd=isnothing(ex) ? NaN : ex.sd,
                     exact_sec=isnothing(ex) ? NaN : ex.sec,
                     exact_events=isnothing(ex) ? NaN : ex.nevents))
        write_csv(joinpath(RESULTS, "e27_hybrid_cost.csv"), rows)
    end
end
