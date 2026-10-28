## E05: per-event diagnostics and efficiency on seir_trees[1] (72 nodes, 37 samples)
## for the POMP-based filters and FastSEIR configurations, single-threaded.
include("common.jl")
using PhyloPOMP: FastSEIR

const R = parse(Int, get(ENV, "E05_R", "30"))
const NP = parse(Int, get(ENV, "E05_NP", "1000"))
const TREE = parse(Int, get(ENV, "E05_TREE", "1"))
const TIME = parse(Float64, get(ENV, "E05_TIME", "50.0"))
Random.seed!(5)

g = parse_newick(GuidedSEIR.seir_trees[TREE], time=TIME)
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
types = [n.type for n ∈ g.nodes]
issample = types .== PhyloPOMP.Sample
isbranch = types .== PhyloPOMP.Node

rows = NamedTuple[]
perevent = NamedTuple[]

record!(label, ll, times, allocs, ess, zf) = begin
    s = summarize(ll)
    tm = mean(times)
    row = (method=label, Np=NP, R=R, lme=s.lme, se=s.se, sd=s.sd, nfinite=s.nfinite,
           sec_per_run=tm, MiB_per_run=mean(allocs)/2^20,
           mean_ess=mean(ess), min_ess=minimum(ess),
           zero_sample=mean(zf[issample]), zero_branch=mean(zf[isbranch]),
           sec_to_sd1=tm*s.sd^2)
    push!(rows, row)
    @printf("%-28s lme=% .2f ± %.2f sd=%.2f t=%.3fs %.0fMiB ESS(mean/min)=%.0f/%.1f zero(sample/branch)=%.3f/%.3f t@sd1=%.3fs\n",
            label, row.lme, row.se, row.sd, row.sec_per_run, row.MiB_per_run, row.mean_ess,
            row.min_ess, row.zero_sample, row.zero_branch, row.sec_to_sd1)
end

## --- POMP-based filters
for v ∈ (:naive, :soft, :guided, :hard)
    p = make_pomp(v, g; par...)
    pfilter(p; Np=50)
    ll = Float64[]; times = Float64[]; allocs = Float64[]
    ess = zeros(length(g)); zf = zeros(length(g))
    for r ∈ 1:R
        stats = @timed pfilter(p; Np=NP)
        pf = stats.value
        push!(ll, logLik(pf)); push!(times, stats.time); push!(allocs, stats.bytes)
        ess .+= POMP.eff_sample_size(pf) ./ R
        zf .+= vec(mean(isinf.(pf.logweights), dims=2)) ./ R
    end
    record!("pomp/$(v)", ll, times, allocs, ess, zf)
    for k ∈ eachindex(ess)
        push!(perevent, (method="pomp/$(v)", event=k, type=string(types[k]), ess=ess[k], zero=zf[k]))
    end
end

## --- FastSEIR
gd = guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!)
tr = FastSEIR.Tree(g, gd)
fastconfigs = [
    ("fast/naive/pre",             FastSEIR.Options(proposal=:naive, guided=false, order=:pre,  threaded=false)),
    ("fast/fixed/pre",             FastSEIR.Options(proposal=:fixed, guided=false, order=:pre,  threaded=false)),
    ("fast/local/pre",             FastSEIR.Options(proposal=:local, guided=false, order=:pre,  threaded=false)),
    ("fast/local/post",            FastSEIR.Options(proposal=:local, guided=false, order=:post, threaded=false)),
    ("fast/fixed/post/guided",     FastSEIR.Options(proposal=:fixed, guided=true,  order=:post, threaded=false)),
    ("fast/local/pre/guided",      FastSEIR.Options(proposal=:local, guided=true,  order=:pre,  threaded=false)),
    ("fast/local/post/guided",     FastSEIR.Options(proposal=:local, guided=true,  order=:post, threaded=false)),
    ("fast/local/post/guided/ess", FastSEIR.Options(proposal=:local, guided=true,  order=:post, threaded=false,
                                                    resample=:ess, trigger=0.5)),
]
for (label, opt) ∈ fastconfigs
    FastSEIR.pfilter(tr, par; Np=50, opt)
    ll = Float64[]; times = Float64[]; allocs = Float64[]
    ess = zeros(length(g)); zf = zeros(length(g))
    for r ∈ 1:R
        stats = @timed FastSEIR.pfilter(tr, par; Np=NP, opt, diagnostics=true)
        res = stats.value
        push!(ll, res.logLik); push!(times, stats.time); push!(allocs, stats.bytes)
        ess .+= replace(res.ess, NaN=>NP) ./ R
        zf .+= replace(res.zerofrac, NaN=>0.0) ./ R
    end
    record!(label, ll, times, allocs, ess, zf)
    for k ∈ eachindex(ess)
        push!(perevent, (method=label, event=k, type=string(types[k]), ess=ess[k], zero=zf[k]))
    end
end

tag = "tree$(TREE)_np$(NP)"
write_csv(joinpath(RESULTS, "e05_summary_$(tag).csv"), rows)
write_csv(joinpath(RESULTS, "e05_perevent_$(tag).csv"), perevent)
println("wrote e05_summary_$(tag).csv and e05_perevent_$(tag).csv")
