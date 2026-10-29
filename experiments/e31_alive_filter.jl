## E31: can alive SMC prevent collapse without losing likelihood precision?
include("common.jl")
include("alive_filter.jl")
using .AliveFilter: pfilter_alive, TrialLimit

const PARTS = get(ENV,"E31_PARTS","ABCD")
const EXACT_REPS = parse(Int,get(ENV,"E31_EXACT_REPS","64"))
const EXACT_NP = parse(Int,get(ENV,"E31_EXACT_NP","1000"))
const REPS = parse(Int,get(ENV,"E31_REPS","24"))
const NP = parse(Int,get(ENV,"E31_NP","1000"))
const LARGE_NP = parse(Int,get(ENV,"E31_LARGE_NP","128"))
const RARE_REPS = parse(Int,get(ENV,"E31_RARE_REPS","512"))
const CAP_MULTIPLIER = parse(Int,get(ENV,"E31_CAP_MULTIPLIER","20"))
const POPS = parse.(Int,split(get(ENV,"E31_POPS","1000,10000"),","))
@assert min(EXACT_REPS,REPS,RARE_REPS) ≥ 2
@assert min(EXACT_NP,NP,LARGE_NP,CAP_MULTIPLIER) > 0 && all(>(0),POPS)

function run_once(tr, θ, method, Np, seed; guided, cap_multiplier)
    try
        result = if method == "systematic"
            pf = pfilter(tr; Np,θ...,guided,rng=Xoshiro(seed))
            (pf=pf,trials=nothing,proposed=nothing)
        else
            pfilter_alive(tr; Np,θ...,guided,alive=method == "alive",
                          max_trials=cap_multiplier*(Np+1),rng=Xoshiro(seed))
        end
        pf = result.pf
        @assert isfinite(logLik(pf)) || logLik(pf) == -Inf "invalid numerical likelihood"
        (status="complete",loglik=logLik(pf),nevents=pf.nevents,
         trials=isnothing(result.trials) ? NaN : sum(result.trials),
         proposed=isnothing(result.proposed) ? NaN : sum(result.proposed),
         cap_interval=0,cap_trials=0,cap_successes=0)
    catch e
        e isa TrialLimit || rethrow()
        (status="capped",loglik=NaN,nevents=e.nevents,trials=e.total_trials,
         proposed=e.total_proposed,cap_interval=e.interval,cap_trials=e.trials,
         cap_successes=e.successes)
    end
end

function metrics(rows)
    ncapped = count(r -> r.status == "capped",rows)
    ll = [r.loglik for r ∈ rows]
    nfinite = count(isfinite,ll)
    nzero = count(==(-Inf),ll)
    sec = mean(r.sec for r ∈ rows)
    if ncapped > 0
        ## A capped attempt has no likelihood estimate. Do not condition the
        ## precision calculation on the subset of runs that finished.
        return (lme=NaN,se=NaN,mean=NaN,sd=NaN,cv2=NaN,variance_cost=NaN,
                max_share=NaN,nfinite,nzero,ncapped,sec)
    elseif nfinite == 0
        return (lme=-Inf,se=Inf,mean=-Inf,sd=Inf,cv2=Inf,variance_cost=Inf,
                max_share=NaN,nfinite,nzero,ncapped,sec)
    end
    a = exp.(ll .- maximum(ll))
    cv2 = var(a)/mean(a)^2
    (lme=maximum(ll)+log(mean(a)),se=sqrt(cv2/length(ll)),mean=mean(ll),
     sd=nzero == 0 ? std(ll) : Inf,cv2,variance_cost=cv2*sec,
     max_share=maximum(a)/sum(a),nfinite,nzero,ncapped,sec)
end

function measure_case(case, g, θ; Np=NP, reps=REPS, guides=("default",),
                      exact=nothing, seed=310000,
                      cap_multiplier=isnothing(exact) ? CAP_MULTIPLIER : max(100,CAP_MULTIPLIER))
    methods = ("systematic","multinomial","alive")
    configs = [(method=m,guide=d,tr=d == "none" ? FastSEIR.Tree(g) :
                FastSEIR.Tree(g,default_guide())) for d ∈ guides for m ∈ methods]
    for c ∈ configs
        run_once(c.tr,θ,c.method,Np,seed; guided=c.guide != "none",cap_multiplier)
    end
    raw = NamedTuple[]
    summary = NamedTuple[]
    for r ∈ 1:reps
        for i ∈ shuffle(Xoshiro(seed+r),collect(eachindex(configs)))
            c = configs[i]
            run_seed = seed+10000+r
            result = @timed run_once(c.tr,θ,c.method,Np,run_seed;
                                     guided=c.guide != "none",cap_multiplier)
            push!(raw,(case=case,method=c.method,guide=c.guide,Np=Np,
                       replicate=r,seed=run_seed,sec=result.time,bytes=result.bytes,
                       result.value...))
        end
        write_csv(joinpath(RESULTS,"e31_$(case)_raw.csv"),raw)
        (r % 8 == 0 || r == reps) && println("$case: replicate $r/$reps")
    end
    for c ∈ configs
        rows = filter(x -> x.method == c.method && x.guide == c.guide,raw)
        m = metrics(rows)
        relative_mean, relative_se, z = if isnothing(exact)
            (NaN,NaN,NaN)
        else
            @assert m.ncapped == 0 "exact-reference cell exceeded its trial cap"
            a = exp.([x.loglik-exact for x ∈ rows])
            avg, se = mean(a), std(a)/sqrt(reps)
            @assert abs(avg-1) < 5se+0.01 "$case $(c.method) $(c.guide) disagrees with exact"
            (avg,se,se == 0 ? 0.0 : (avg-1)/se)
        end
        @printf("%-10s %-11s %-7s Np=%4d SD=%6.2f lme=%8.2f +/- %.2f %.3fs finite=%d/%d zero=%d capped=%d share=%.2f\n",
                case,c.method,c.guide,Np,m.sd,m.lme,m.se,m.sec,
                m.nfinite,reps,m.nzero,m.ncapped,m.max_share)
        push!(summary,(case=case,method=c.method,guide=c.guide,Np=Np,reps=reps,
                       cap_multiplier,exact=isnothing(exact) ? NaN : exact,
                       relative_mean,relative_se,z,m...,
                       mean_events=mean(r.nevents for r ∈ rows),
                       mean_trials=mean(r.trials for r ∈ rows),
                       mean_proposed=mean(r.proposed for r ∈ rows)))
    end
    write_csv(joinpath(RESULTS,"e31_$(case)_summary.csv"),summary)
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
        measure_case(c.case,c.g,c.θ; Np=EXACT_NP,reps=EXACT_REPS,
                     guides=("none","default"),exact,seed=311000+100000i)
    end
    rare_g = parse_newick("([&&PhyloPOMP type=sample]:1.0);",time=1.0)
    rare_θ = (β=0.0,σ=0.0,γ=0.0,ω=0.0,ψ=0.4,χ=0.0,
              pop=2,S0=0.0,E0=0.5,I0=0.5,R0=0.0)
    measure_case("rare_root",rare_g,rare_θ; Np=2,reps=RARE_REPS,guides=("none",),
                 exact=ExactSEIR.loglik(rare_g; rare_θ...),seed=315000)
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
    measure_case("E19_fit",g,θ; seed=316000)
    g2 = parse_newick(FastSEIR.seir_trees[2])
    θ2 = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    measure_case("tree2_fit",g2,θ2; seed=317000)
end

if occursin('C',PARTS)
    for N ∈ POPS
        local θ = merge(DEFAULTS,(pop=N,ψ=8/N))
        local g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
        println("N=$N: $(g.nsample) samples")
        measure_case("N$N",g,θ; Np=LARGE_NP,guides=("none","default"),seed=318000+N)
    end
end

if occursin('D',PARTS)
    θ = merge(DEFAULTS,(pop=10000,ψ=8/10000))
    g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
    measure_case("N10000_highcap",g,θ; Np=LARGE_NP,guides=("none",),
                 cap_multiplier=1000,seed=329000)
end
