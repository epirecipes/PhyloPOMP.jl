## E13: likelihood slices for β on a simulated genealogy with known parameters,
## evaluated with independent random numbers or with common random numbers (CRN).
## CRN: replicate r uses the same master seed at every β, so that the estimates
## at nearby β are positively correlated and their differences precise.
include("common.jl")

const R = parse(Int, get(ENV, "E13_R", "10"))
const NP = parse(Int, get(ENV, "E13_NP", "2000"))
const T = parse(Float64, get(ENV, "E13_T", "50"))

truth = DEFAULTS
sim = SimSEIR.simulate(; truth..., time=T, rng=Xoshiro(2026))
g = sim.genealogy
@printf("simulated genealogy: %d samples, %d nodes, %d population events\n",
        g.nsample, length(g), sim.nevents)
tr = FastSEIR.Tree(g, matched_guide(truth))

βs = collect(range(2.5, 6.5, length=17))
## ll[s][i,r]: replicate r at βs[i] under scheme s
schemes = (independent = (i, r) -> Xoshiro(hash((i, r, :independent))),
           crn = (i, r) -> Xoshiro(r))
ll = Dict{Symbol,Matrix{Float64}}()
secs = Dict{Symbol,Float64}()
for (name, rngof) ∈ pairs(schemes)
    L = zeros(length(βs), R)
    secs[name] = @elapsed for i ∈ eachindex(βs), r ∈ 1:R
        L[i,r] = logLik(pfilter(tr; Np=NP, truth..., β=βs[i], rng=rngof(i, r)))
    end
    ll[name] = L
end

## quadratic fit of the per-β logmeanexp; returns MLE and curvature-based SE
quadfit(x, y) = begin
    X = hcat(ones(length(x)), x, x.^2)
    c = X \ y
    β̂ = -c[2]/(2c[3])
    (mle = β̂, se = sqrt(-1/(2c[3])), coef = c)
end

rows = NamedTuple[]
summary = NamedTuple[]
for (name, L) ∈ ll
    lme = [logmeanexp(L[i,:]) for i ∈ eachindex(βs)]
    ## Monte Carlo SD of the difference between adjacent grid points, and between
    ## each point and the grid point nearest the truth, across replicates
    iref = argmin(abs.(βs .- truth.β))
    sd_adj = mean(std(L[i+1,:] .- L[i,:]) for i ∈ 1:length(βs)-1)
    sd_ref = mean(std(L[i,:] .- L[iref,:]) for i ∈ eachindex(βs) if i ≠ iref)
    sd_pt = mean(std(L[i,:]) for i ∈ eachindex(βs))
    q = quadfit(βs, lme)
    ## MLE spread across replicates (each replicate's own quadratic fit)
    mles = [quadfit(βs, L[:,r]).mle for r ∈ 1:R]
    @printf("%-12s sd(point)=%.3f  sd(adjacent diff)=%.3f  sd(diff to β≈truth)=%.3f  MLE=%.2f (curv. SE %.2f; truth %.1f)  sd(MLE over replicates)=%.3f  time=%.0fs\n",
            name, sd_pt, sd_adj, sd_ref, q.mle, q.se, truth.β, std(mles), secs[name])
    push!(summary, (scheme=String(name), Np=NP, R=R, sd_point=sd_pt, sd_adjacent=sd_adj,
                    sd_to_truth=sd_ref, mle=q.mle, mle_se=q.se, sd_mle_replicates=std(mles),
                    seconds=secs[name]))
    for i ∈ eachindex(βs), r ∈ 1:R
        push!(rows, (scheme=String(name), β=βs[i], replicate=r, logLik=L[i,r]))
    end
end
write_csv(joinpath(RESULTS, "e13_crn_slices.csv"), rows)
write_csv(joinpath(RESULTS, "e13_crn_summary.csv"), summary)
