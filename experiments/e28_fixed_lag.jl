## E28: fixed-lag Fisher scores. The likelihood and random trajectories must
## match the path-space filter exactly; only the sufficient statistics change.
include("common.jl")

const PARTS = get(ENV, "E28_PARTS", "AB")
const REPS = parse(Int, get(ENV, "E28_REPS", "32"))
const REF_REPS = parse(Int, get(ENV, "E28_REF_REPS", "24"))
const REF_NP = parse(Int, get(ENV, "E28_REF_NP", "64000"))
const NPS = parse.(Int, split(get(ENV, "E28_NPS", "1000,4000"), ","))
const LAGS = parse.(Int, split(get(ENV, "E28_LAGS", "0,1,5,10,20,50,100"), ","))
const RATES = (:β,:σ,:γ,:ω,:ψ)
@assert REPS ≥ 2 && REF_REPS ≥ 2 && REF_NP > 0
@assert all(>(0),NPS) && all(≥(0),LAGS) && allunique(LAGS)

score_run(tr, θ, Np, lag, seed) = begin
    result = @timed pfilter(tr; Np, θ..., smoothing_lag=lag, rng=Xoshiro(seed))
    pf = result.value
    isfinite(logLik(pf)) || error("filter collapsed: Np=$Np lag=$lag seed=$seed")
    (pf=pf, score=FastSEIR.score(pf; θ...), sec=result.time, bytes=result.bytes)
end

function reference_score(tr, θ, seed)
    samples = [score_run(tr,θ,REF_NP,nothing,seed+r) for r ∈ 1:REF_REPS]
    means = NamedTuple{RATES}(map(x -> mean(s.score[x] for s ∈ samples),RATES))
    ses = NamedTuple{RATES}(map(x -> std([s.score[x] for s ∈ samples])/sqrt(REF_REPS),RATES))
    means, ses, samples
end

function compare_scores(case, tr, θ, reference, reference_se;
                        nps=NPS, lags=LAGS, reps=REPS, seed=280000)
    methods = Union{Nothing,Int}[nothing; lags]
    raw = NamedTuple[]
    summary = NamedTuple[]
    for lag ∈ methods
        score_run(tr,θ,minimum(nps),lag,seed)   # compile outside the measured runs
    end
    for Np ∈ nps
        for r ∈ 1:reps
            results = map(shuffle(Xoshiro(seed+r),methods)) do lag
                run = score_run(tr,θ,Np,lag,seed+1000+r)
                label = isnothing(lag) ? "path" : string(lag)
                push!(raw, (case=case, Np=Np, lag=label, replicate=r, seed=seed+1000+r,
                            loglik=logLik(run.pf), sec=run.sec, bytes=run.bytes, run.score...))
                (lag=lag,pf=run.pf)
            end
            baseline = only(x.pf for x ∈ results if isnothing(x.lag))
            for (;lag,pf) ∈ results
                @assert logLik(pf) == logLik(baseline)
                @assert pf.cond_logLik == baseline.cond_logLik && pf.nevents == baseline.nevents
                if !isnothing(lag) && lag ≥ length(tr.time)-1
                    @assert pf.suffstats == baseline.suffstats
                end
            end
            write_csv(joinpath(RESULTS,"e28_$(case)_raw.csv"),raw)
        end
        for lag ∈ methods
            label = isnothing(lag) ? "path" : string(lag)
            rows = filter(x -> x.Np == Np && x.lag == label,raw)
            for rate ∈ RATES
                values = getproperty.(rows,rate)
                avg, sd = mean(values), std(values)
                bias = avg-reference[rate]
                push!(summary, (case=case, Np=Np, lag=label, rate=string(rate), reps=reps,
                                mean=avg, sd=sd, se=sd/sqrt(reps), reference=reference[rate],
                                reference_se=reference_se[rate], bias=bias,
                                difference_se=hypot(sd/sqrt(reps),reference_se[rate]),
                                rmse_proxy=hypot(sd,bias), sec=mean(x.sec for x ∈ rows),
                                bytes=mean(x.bytes for x ∈ rows)))
            end
            βrow = summary[end-4]
            @printf("%-13s Np=%5d lag=%4s  beta score %7.2f +/- %5.2f  bias=%7.2f  RMSE~%6.2f  %.3fs\n",
                    case,Np,label,βrow.mean,βrow.sd,βrow.bias,βrow.rmse_proxy,βrow.sec)
        end
        write_csv(joinpath(RESULTS,"e28_$(case)_summary.csv"),summary)
    end
    summary
end

if occursin('A',PARTS)
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    θ = (β=4.0, σ=0.8, γ=0.5, ω=0.2, ψ=0.4, χ=0.0,
         pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7)
    derivative(rate,h) = begin
        plus = merge(θ,NamedTuple{(rate,)}((θ[rate]*exp(h),)))
        minus = merge(θ,NamedTuple{(rate,)}((θ[rate]*exp(-h),)))
        (ExactSEIR.loglik(g; plus...)-ExactSEIR.loglik(g; minus...))/(2h)
    end
    exact = NamedTuple{RATES}(map(x -> derivative(x,1e-4),RATES))
    for rate ∈ RATES
        @assert isapprox(exact[rate],derivative(rate,5e-5); atol=1e-5)
    end
    tr = FastSEIR.Tree(g,default_guide())
    println("Exact log-rate score on tiny genealogy: ",exact)
    rows = compare_scores("tiny",tr,θ,exact,map(_ -> 0.0,exact);
                          nps=[4000],lags=[0,1,2,length(tr.time)],seed=281000)
    for row ∈ filter(x -> x.lag == "path",rows)
        @assert abs(row.bias) < 5row.se+0.02 "path score disagrees with exact derivative"
    end
end

if occursin('B',PARTS)
    truth = merge(DEFAULTS,(ψ=0.05,))
    g = let rng=Xoshiro(1919), sim=nothing
        while true
            sim = SimSEIR.simulate(; truth...,time=60.0,rng)
            80 ≤ sim.genealogy.nsample ≤ 160 && break
        end
        sim.genealogy
    end
    tr = FastSEIR.Tree(g,default_guide())
    println("E19 genealogy: $(g.nsample) samples, $(length(tr.time)) events")
    mle = merge(truth,(β=3.962,γ=1.048,ψ=0.0690))
    for (j,(case,θ)) ∈ enumerate((("mle",mle),("beta_low",merge(mle,(β=2.8,)))))
        ref, se, runs = reference_score(tr,θ,282000+10000j)
        write_csv(joinpath(RESULTS,"e28_$(case)_reference.csv"),
                  [(replicate=r,Np=REF_NP,loglik=logLik(s.pf),sec=s.sec,s.score...) for (r,s) ∈ enumerate(runs)])
        println("$case reference score: $ref (SE $se)")
        compare_scores(case,tr,θ,ref,se; seed=283000+10000j)
    end
end

if occursin('C',PARTS)
    tr = FastSEIR.Tree(parse_newick(FastSEIR.seir_trees[2]),default_guide())
    θ = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    ref, se, runs = reference_score(tr,θ,286000)
    write_csv(joinpath(RESULTS,"e28_tree2_reference.csv"),
              [(replicate=r,Np=REF_NP,loglik=logLik(s.pf),sec=s.sec,s.score...) for (r,s) ∈ enumerate(runs)])
    println("tree 2 reference score: $ref (SE $se)")
    compare_scores("tree2",tr,θ,ref,se; seed=287000)
end
