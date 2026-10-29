## E26: common random numbers with the coupled filter (experiments/coupled_filter.jl).
## Part A: exactness against ExactSEIR.
## Part B: correlation of log-likelihood estimates at nearby β with a shared seed.
## Part C: β slices (as in E13) and the spread of their maximizers.
## Part D: variance of the estimate with particles resampled in state order.
include("common.jl")
include("coupled_filter.jl")
using .Coupled: pfilter_coupled, pfilter_ordered

const PARTS = get(ENV, "E26_PARTS", "ABC")
const NP = parse(Int, get(ENV, "E26_NP", "1000"))
const RB = parse(Int, get(ENV, "E26_RB", "30"))
const RC = parse(Int, get(ENV, "E26_RC", "10"))

methods = [
    ("fast, shared rng", (tr, θ, r) -> logLik(pfilter(tr; Np=NP, θ..., rng=Xoshiro(r)))),
    ("coupled, slot order", (tr, θ, r) -> logLik(pfilter_coupled(tr; Np=NP, θ..., sorted=false, seed=UInt64(r)))),
    ("coupled, state order", (tr, θ, r) -> logLik(pfilter_coupled(tr; Np=NP, θ..., sorted=true, seed=UInt64(r)))),
]

## ---------------- Part A: exactness
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
    for c ∈ cases
        exact = ExactSEIR.loglik(c.g; c.par...)
        tr = FastSEIR.Tree(c.g, default_guide())
        ll = [logLik(pfilter_coupled(tr; Np=10000, c.par..., seed=UInt64(r))) for r ∈ 1:40]
        d, se = mean(ll)-exact, std(ll)/sqrt(40)
        @printf("== %-20s exact=%.4f  coupled: mean-exact=% .4f ± %.4f  z=% .2f\n", c.name, exact, d, se, d/se)
        push!(rowsA, (case=c.name, exact=exact, bias=d, se=se, z=d/se))
    end
    write_csv(joinpath(RESULTS, "e26_coupled_exactness.csv"), rowsA)
end

## the genealogy of E19 (92 samples) and its generating parameters
truth = merge(DEFAULTS, (ψ=0.05,))
g19 = let rng = Xoshiro(1919), sim = nothing
    while true
        sim = SimSEIR.simulate(; truth..., time=60.0, rng)
        80 ≤ sim.genealogy.nsample ≤ 160 && break
    end
    sim.genealogy
end
tr19 = FastSEIR.Tree(g19, default_guide())

## ---------------- Part B: correlation at nearby β
if occursin('B', PARTS)
    rowsB = NamedTuple[]
    for (label, f) ∈ methods
        f(tr19, truth, 0)
        t = @elapsed base = [f(tr19, truth, r) for r ∈ 1:RB]
        for δ ∈ (0.01, 0.03, 0.1)
            near = [f(tr19, merge(truth, (β = truth.β*(1+δ),)), r) for r ∈ 1:RB]
            ρ = cor(base, near)
            @printf("%-22s δ=%.2f  corr=%.3f  sd(diff)=%.3f  sd=%.3f  [%.2fs/run]\n",
                    label, δ, ρ, std(near .- base), std(base), t/RB)
            push!(rowsB, (method=label, δ=δ, corr=ρ, sd_diff=std(near .- base), sd=std(base), sec=t/RB))
        end
    end
    write_csv(joinpath(RESULTS, "e26_coupled_correlation.csv"), rowsB)
end

## ---------------- Part C: β slices
if occursin('C', PARTS)
    βs = collect(range(2.5, 6.5, length=17))
    quadmax(x, y) = (c = hcat(ones(length(x)), x, x.^2) \ y; -c[2]/(2c[3]))
    rowsC = NamedTuple[]
    slices = NamedTuple[]
    for (label, f, shared) ∈ (("fast, independent", methods[1][2], false),
                              ("fast, shared rng", methods[1][2], true),
                              ("coupled, state order", methods[3][2], true))
        L = [f(tr19, merge(truth, (β=β,)), shared ? r : 1000r + i) for (i, β) ∈ enumerate(βs), r ∈ 1:RC]
        sd_adj = mean(std(L[i+1,:] .- L[i,:]) for i ∈ 1:length(βs)-1)
        mles = [quadmax(βs, L[:,r]) for r ∈ 1:RC]
        pooled = quadmax(βs, [logmeanexp(L[i,:]) for i ∈ eachindex(βs)])
        @printf("%-22s sd(point)=%.3f  sd(adjacent diff)=%.3f  pooled MLE=%.2f  sd(replicate MLEs)=%.3f\n",
                label, mean(std(L[i,:]) for i ∈ eachindex(βs)), sd_adj, pooled, std(mles))
        push!(rowsC, (method=label, sd_point=mean(std(L[i,:]) for i ∈ eachindex(βs)), sd_adjacent=sd_adj,
                      mle=pooled, sd_mle=std(mles)))
        append!(slices, [(method=label, β=βs[i], replicate=r, logLik=L[i,r]) for i ∈ eachindex(βs), r ∈ 1:RC])
    end
    write_csv(joinpath(RESULTS, "e26_coupled_slices_summary.csv"), rowsC)
    write_csv(joinpath(RESULTS, "e26_coupled_slices.csv"), slices)
end

## ---------------- Part D: resampling in state order
if occursin('D', PARTS)
    fitted = merge(DEFAULTS, (β=2.806, σ=0.599, γ=0.478, ω=5.171, ψ=0.0271, pop=173))
    rowsD = NamedTuple[]
    cases = [
        ("E19 tree", tr19, truth, 1000, 40),
        ("seir_trees[1]", FastSEIR.Tree(parse_newick(FastSEIR.seir_trees[1], time=50.0), default_guide()), DEFAULTS, 1000, 40),
        ("seir_trees[2] (E21b estimate)", FastSEIR.Tree(parse_newick(FastSEIR.seir_trees[2]), default_guide()), fitted, 4000, 16),
    ]
    for (name, tr, θ, Np, R) ∈ cases
        for (label, f) ∈ (("fast", (r) -> pfilter(tr; Np, θ..., rng=Xoshiro(r))),
                          ("fast, state order", (r) -> pfilter_ordered(tr; Np, θ..., rng=Xoshiro(r))))
            f(0)
            res = [(@timed logLik(f(r))) for r ∈ 1:R]
            ll = [x.value for x ∈ res]
            sec = mean(x.time for x ∈ res)
            @printf("%-30s %-18s Np=%d: mean=%.2f sd=%.3f  %.2fs/run  t@sd1=%.3fs\n",
                    name, label, Np, mean(ll), std(ll), sec, sec*std(ll)^2)
            push!(rowsD, (tree=name, method=label, Np=Np, R=R, mean=mean(ll), sd=std(ll), sec=sec,
                          sec_to_sd1=sec*std(ll)^2))
        end
        write_csv(joinpath(RESULTS, "e26_ordered_resampling.csv"), rowsD)
    end
end
