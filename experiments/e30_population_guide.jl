## E30: forward-population forecast colour guidance, with exact target weights.
include("common.jl")
include("population_guide.jl")
using .PopulationGuide: ForecastTree, pfilter_forecast

const PARTS = get(ENV,"E30_PARTS","ABC")
const EXACT_REPS = parse(Int,get(ENV,"E30_EXACT_REPS","64"))
const EXACT_NP = parse(Int,get(ENV,"E30_EXACT_NP","2000"))
const REPS = parse(Int,get(ENV,"E30_REPS","32"))
const NP = parse(Int,get(ENV,"E30_NP","1000"))
const LARGE_NP = parse(Int,get(ENV,"E30_LARGE_NP","512"))
const POPS = parse.(Int,split(get(ENV,"E30_POPS","1000,10000"),","))
@assert min(EXACT_REPS,REPS) ≥ 2 && min(EXACT_NP,NP,LARGE_NP) > 0
@assert all(>(0),POPS)

prepare(g, θ, method) = if method == "default"
    FastSEIR.Tree(g,default_guide())
elseif method == "equilibrium"
    FastSEIR.Tree(g,matched_guide(θ))
elseif method == "forecast"
    ForecastTree(g; θ...)
elseif method == "forecast_fine"
    ForecastTree(g; θ...,step=0.005)
else
    error("unknown guide $method")
end

run_filter(tr::FastSEIR.Tree, θ, Np, seed) = pfilter(tr; Np,θ...,rng=Xoshiro(seed))
run_filter(tr::ForecastTree, θ, Np, seed) = pfilter_forecast(tr; Np,rng=Xoshiro(seed))

function metrics(rows)
    ll = [r.loglik for r ∈ rows]
    nfinite = count(isfinite,ll)
    sec = mean(r.sec for r ∈ rows)
    if nfinite == 0
        return (lme=-Inf,se=Inf,mean=-Inf,sd=Inf,cv2=Inf,variance_cost=Inf,
                max_share=NaN,nfinite=0,sec=sec)
    end
    a = exp.(ll .- maximum(ll))
    cv2 = var(a)/mean(a)^2
    (lme=maximum(ll)+log(mean(a)),se=sqrt(cv2/length(ll)),mean=mean(ll),
     sd=nfinite == length(ll) ? std(ll) : Inf,cv2=cv2,variance_cost=cv2*sec,
     max_share=maximum(a)/sum(a),nfinite=nfinite,sec=sec)
end

function measure_case(case, g, θ; Np=NP, reps=REPS, exact=nothing, seed=300000)
    methods = isnothing(exact) ? ("default","equilibrium","forecast") :
        ("default","equilibrium","forecast","forecast_fine")
    ## Warm compilation separately from measured guide construction.
    for method ∈ methods
        tr = prepare(g,θ,method)
        run_filter(tr,θ,Np,seed)
    end
    compiled = [(@timed prepare(g,θ,m)) for m ∈ methods]
    raw = NamedTuple[]
    summary = NamedTuple[]
    for r ∈ 1:reps
        for i ∈ shuffle(Xoshiro(seed+r),collect(eachindex(methods)))
            run_seed = seed+10000+r
            result = @timed run_filter(compiled[i].value,θ,Np,run_seed)
            pf = result.value
            @assert isfinite(logLik(pf)) || logLik(pf) == -Inf "invalid numerical likelihood"
            push!(raw,(case=case,method=methods[i],Np=Np,replicate=r,seed=run_seed,
                       loglik=logLik(pf),sec=result.time,bytes=result.bytes,
                       nevents=pf.nevents,ess_frac=mean(eff_sample_size(pf))/Np))
        end
        write_csv(joinpath(RESULTS,"e30_$(case)_raw.csv"),raw)
        (r % 8 == 0 || r == reps) && println("$case: replicate $r/$reps")
    end
    for (i,method) ∈ enumerate(methods)
        rows = filter(x -> x.method == method,raw)
        m = metrics(rows)
        relative_mean, relative_se, z = if isnothing(exact)
            (NaN,NaN,NaN)
        else
            a = exp.([x.loglik-exact for x ∈ rows])
            avg, se = mean(a), std(a)/sqrt(reps)
            @assert abs(avg-1) < 5se+0.01 "$case $method disagrees with exact likelihood"
            (avg,se,(avg-1)/se)
        end
        @printf("%-13s %-13s Np=%4d mean=%8.2f SD=%6.2f lme=%8.2f +/- %.2f %.3fs  finite=%d/%d max_share=%.2f\n",
                case,method,Np,m.mean,m.sd,m.lme,m.se,m.sec,m.nfinite,reps,m.max_share)
        push!(summary,(case=case,method=method,Np=Np,reps=reps,
                       exact=isnothing(exact) ? NaN : exact,
                       relative_mean=relative_mean,relative_se=relative_se,z=z,m...,
                       build_sec=compiled[i].time,build_bytes=compiled[i].bytes))
    end
    write_csv(joinpath(RESULTS,"e30_$(case)_summary.csv"),summary)
    summary
end

if occursin('A',PARTS)
    base = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0)
    cases = (
        (case="single",g=parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);",time=1.8),
         θ=merge(base,(σ=0.7,pop=6,S0=4/6,E0=0.0,I0=1/6,R0=1/6))),
        (case="branch",g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6),
         θ=merge(base,(pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7))),
        (case="root_EI",g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6),
         θ=merge(base,(pop=10,S0=0.7,E0=0.1,I0=0.1,R0=0.1))),
    )
    for (i,c) ∈ enumerate(cases)
        exact = ExactSEIR.loglik(c.g; c.θ...)
        measure_case(c.case,c.g,c.θ; Np=EXACT_NP,reps=EXACT_REPS,exact,seed=301000+100000i)
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
    θ = merge(truth,(β=3.962,γ=1.048,ψ=0.0690))
    measure_case("E19_fit",g,θ; seed=302000)
    g2 = parse_newick(FastSEIR.seir_trees[2])
    θ2 = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    measure_case("tree2_fit",g2,θ2; seed=303000)
end

if occursin('C',PARTS)
    for N ∈ POPS
        local θ = merge(DEFAULTS,(pop=N,ψ=8/N))
        local g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
        println("N=$N: $(g.nsample) samples")
        measure_case("N$N",g,θ; Np=LARGE_NP,seed=304000+N)
    end
end
