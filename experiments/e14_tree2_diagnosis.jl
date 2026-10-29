## E14: where does the filter degenerate on the large genealogy seir_trees[2]
## (1562 events, 864 samples, T≈399)?  Per-interval ESS, zero-weight fraction
## and variance of the conditional log likelihoods, related to the length of the
## interval, the type of the closing event and the number of lineages present.
include("common.jl")

const R = parse(Int, get(ENV, "E14_R", "8"))
const NP = parse(Int, get(ENV, "E14_NP", "4000"))
Random.seed!(1414)

g = parse_newick(FastSEIR.seir_trees[2])
par = DEFAULTS
guides = [("default", default_guide()),
          ("πE=0.5 c=4", fsmarkov(Expos=>0.5, Infec=>0.5, (Expos,Infec)=>4))]

M = length(g)
type = [string(g[k].type) for k ∈ 1:M]
nlin = let A = 0
    map(1:M) do k
        A += length(g[k].children) - (g[k].type == PhyloPOMP.Root ? 0 : 1)
        A
    end
end
len = [(k < M ? g[k+1].slate : g.time) - g[k].slate for k ∈ 1:M]
closing = [k < M ? type[k+1] : "end" for k ∈ 1:M]

summary = NamedTuple[]
perint = NamedTuple[]
for (label, m) ∈ guides
    tr = FastSEIR.Tree(g, m)
    pfilter(tr; Np=100, par...)
    ll = zeros(R); ess = zeros(M, R); cll = zeros(M, R); zf = zeros(M, R); t = zeros(R)
    for r ∈ 1:R
        t[r] = @elapsed pf = pfilter(tr; Np=NP, par...)
        ll[r] = logLik(pf)
        ess[:,r] .= eff_sample_size(pf); cll[:,r] .= cond_logLik(pf); zf[:,r] .= pf.zero_fraction
    end
    v = vec(var(cll, dims=2))
    me = vec(mean(ess, dims=2))
    @printf("\n== %s (Np=%d, R=%d): logLik mean %.1f sd %.1f range [%.1f, %.1f], %.1fs/run\n",
            label, NP, R, mean(ll), std(ll), minimum(ll), maximum(ll), mean(t))
    @printf("   intervals with mean ESS < 1%% of Np: %d of %d;  mean zero fraction %.3f\n",
            count(<(0.01NP), me), M, mean(zf))
    for ty ∈ ("Sample", "Node", "end")
        sel = closing .== ty
        @printf("   closing %-6s: %4d intervals, share of Σvar %.2f, mean ESS/Np %.3f, mean zero %.3f\n",
                ty, count(sel), sum(v[sel])/sum(v), mean(me[sel])/NP, mean(zf[sel,:]))
    end
    @printf("   corr(var, interval length) = %.2f, corr(var, lineages) = %.2f\n",
            cor(v, len), cor(v, nlin))
    top = sortperm(v, rev=true)[1:8]
    println("   top intervals (k: closing, length, lineages, var, ESS/Np): ",
            join(["$(k): $(closing[k]), $(round(len[k],digits=2)), $(nlin[k]), $(round(v[k],digits=1)), $(round(me[k]/NP,digits=3))" for k ∈ top], " | "))
    push!(summary, (guide=label, Np=NP, R=R, mean_ll=mean(ll), sd_ll=std(ll), sec=mean(t),
                    collapsed=count(<(0.01NP), me), mean_zero=mean(zf)))
    for k ∈ 1:M
        push!(perint, (guide=label, interval=k, closing=closing[k], length=len[k], lineages=nlin[k],
                       var_cll=v[k], mean_ess=me[k], zero=mean(zf[k,:])))
    end
end
write_csv(joinpath(RESULTS, "e14_tree2_summary.csv"), summary)
write_csv(joinpath(RESULTS, "e14_tree2_intervals.csv"), perint)
