## E07: where does the log-likelihood variance come from? Per-event variance of the
## conditional log-likelihood on seir_trees[1] for the best FastSEIR configurations,
## with R = 100 replicates (single thread).
include("common.jl")
using PhyloPOMP: FastSEIR

const R = parse(Int, get(ENV, "E07_R", "100"))
const NP = parse(Int, get(ENV, "E07_NP", "1000"))
Random.seed!(707)

g = parse_newick(GuidedSEIR.seir_trees[1], time=50.0)
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
tr = FastSEIR.Tree(g, guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!))
types = string.([n.type for n ∈ g.nodes])
lens = tr.tend .- tr.time            # interval following event k
prevlen = [k == 1 ? 0.0 : lens[k-1] for k ∈ eachindex(lens)]   # interval ending at event k

configs = [
    ("local/post/guided",       FastSEIR.Options(threaded=false)),
    ("apf lineage",             FastSEIR.Options(threaded=false, lookahead=:lineage)),
    ("apf both h=1",            FastSEIR.Options(threaded=false, lookahead=:both, horizon=1)),
    ("apf pop h=1",             FastSEIR.Options(threaded=false, lookahead=:pop, horizon=1)),
]
summary = NamedTuple[]
decomp = NamedTuple[]
for (label, opt) ∈ configs
    FastSEIR.pfilter(tr, par; Np=50, opt)
    C = zeros(R, length(g)); ll = zeros(R); tt = zeros(R)
    for r ∈ 1:R
        st = @timed FastSEIR.pfilter(tr, par; Np=NP, opt, diagnostics=true)
        ll[r] = st.value.logLik; tt[r] = st.time
        C[r,:] .= replace(st.value.cll, NaN=>0.0)
    end
    s = summarize(ll)
    v = vec(var(C, dims=1))
    tot = var(ll)
    ## variance shares by event type (events whose weight enters at that resampling step)
    share(t) = sum(v[types .== t])/sum(v)
    @printf("%-20s lme=% .3f ± %.3f sd=%.3f (Σvar_k=%.3f, var=%.3f) t=%.3fs t@sd1=%.3fs  share: Sample=%.2f Node=%.2f Root=%.2f\n",
            label, s.lme, s.se, s.sd, sum(v), tot, mean(tt), mean(tt)*s.sd^2,
            share("Sample"), share("Node"), share("Root"))
    ## correlation of per-event variance with the preceding interval length
    @printf("    corr(var_k, length of interval ending at event k) = %.2f; top-5 events: %s\n",
            cor(v, prevlen), join(["$(k)($(types[k]),Δt=$(round(prevlen[k],digits=2)),v=$(round(v[k],digits=3)))"
                                   for k ∈ sortperm(v, rev=true)[1:5]], " "))
    push!(summary, (config=label, R=R, Np=NP, lme=s.lme, se=s.se, sd=s.sd, sum_var_k=sum(v),
                    sec_per_run=mean(tt), sec_to_sd1=mean(tt)*s.sd^2,
                    share_sample=share("Sample"), share_node=share("Node")))
    for k ∈ eachindex(v)
        push!(decomp, (config=label, event=k, type=types[k], dt_before=prevlen[k], var_cll=v[k]))
    end
end
write_csv(joinpath(RESULTS, "e07_variance_summary.csv"), summary)
write_csv(joinpath(RESULTS, "e07_variance_per_event.csv"), decomp)
