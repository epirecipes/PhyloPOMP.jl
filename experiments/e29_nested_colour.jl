## E29: conditional colour SMC sharing a population trajectory.
include("common.jl")
include("nested_colour_filter.jl")
using .NestedColour: pfilter_nested

const PARTS = get(ENV,"E29_PARTS","AB")
const EXACT_REPS = parse(Int,get(ENV,"E29_EXACT_REPS","64"))
const EXACT_BUDGET = parse(Int,get(ENV,"E29_EXACT_BUDGET","2000"))
const REPS = parse(Int,get(ENV,"E29_REPS","24"))
const BUDGET = parse(Int,get(ENV,"E29_BUDGET","512"))
const POPS = parse.(Int,split(get(ENV,"E29_POPS","1000,10000"),","))
const MODES = split(get(ENV,"E29_MODES","unguided,guided"),",")
@assert EXACT_REPS ≥ 2 && REPS ≥ 2
@assert all(>(0),POPS) && all(x -> x ∈ ("unguided","guided","matched"),MODES)

configs(budget; exact=false) = begin
    @assert budget ≥ 16 && budget % 16 == 0 "colour budget must be divisible by 16"
    choices = [(method="FastSEIR",Np=budget,K=0,check=32)]
    append!(choices,[(method="nested_K$K",Np=budget÷K,K=K,check=32) for K ∈ (1,4,16)])
    exact && push!(choices,(method="nested_K4_check1",Np=budget÷4,K=4,check=1))
    choices
end

run_filter(tr, θ, c, seed) = begin
    if c.K == 0
        pfilter(tr; Np=c.Np,θ...,rng=Xoshiro(seed))
    else
        pfilter_nested(tr; Np=c.Np,K=c.K,inner_check=c.check,θ...,rng=Xoshiro(seed))
    end
end

function metrics(rows)
    ll = [r.loglik for r ∈ rows]
    nfinite = count(isfinite,ll)
    sec = mean(r.sec for r ∈ rows)
    if nfinite == 0
        return (lme=-Inf,se=Inf,mean=-Inf,sd=Inf,cv2=Inf,variance_cost=Inf,
                max_share=NaN,nfinite=0,sec=sec)
    end
    mx = maximum(ll)
    a = exp.(ll .- mx)
    cv2 = var(a)/mean(a)^2
    (lme=mx+log(mean(a)),se=sqrt(cv2/length(ll)),mean=mean(ll),
     sd=nfinite == length(ll) ? std(ll) : Inf,cv2=cv2,variance_cost=cv2*sec,
     max_share=maximum(a)/sum(a),nfinite=nfinite,sec=sec)
end

function measure_case(case, g, θ; budget=BUDGET, reps=REPS, exact=nothing,
                      modes=MODES, seed=290000)
    choices = configs(budget; exact=!isnothing(exact))
    raw = NamedTuple[]
    summary = NamedTuple[]
    for (mode_id,mode) ∈ enumerate(modes)
        tr = mode == "unguided" ? FastSEIR.Tree(g) :
            FastSEIR.Tree(g,mode == "matched" ? matched_guide(θ) : default_guide())
        for (i,c) ∈ enumerate(choices)
            run_filter(tr,θ,c,seed+100mode_id+i)    # warm up outside measurements
        end

        for r ∈ 1:reps
            for c ∈ shuffle(Xoshiro(seed+r),choices)
                run_seed = seed+10000mode_id+r
                result = @timed run_filter(tr,θ,c,run_seed)
                pf = result.value
                @assert isfinite(logLik(pf)) || logLik(pf) == -Inf "invalid numerical likelihood"
                push!(raw,(case=case,mode=mode,method=c.method,Np=c.Np,K=c.K,
                           inner_check=c.check,replicate=r,seed=run_seed,
                           loglik=logLik(pf),sec=result.time,bytes=result.bytes,
                           nevents=pf.nevents,inner_resamples=c.K == 0 ? 0 : pf.inner_resamples))
            end
            write_csv(joinpath(RESULTS,"e29_$(case)_raw.csv"),raw)
            (r % 4 == 0 || r == reps) && println("$case $mode: replicate $r/$reps")
        end
        for c ∈ choices
            rows = filter(x -> x.mode == mode && x.method == c.method,raw)
            m = metrics(rows)
            relative_mean, relative_se, z = if isnothing(exact)
                (NaN,NaN,NaN)
            else
                ratios = exp.([x.loglik-exact for x ∈ rows])
                avg, se = mean(ratios), std(ratios)/sqrt(reps)
                @assert abs(avg-1) < 5se+0.01 "$case $mode $(c.method) disagrees with exact likelihood"
                (avg,se,(avg-1)/se)
            end
            @printf("%-10s %-8s %-19s G=%4d K=%2d  logmean=%8.2f +/- %.2f mean=%8.2f SD=%6.2f %.3fs  finite=%d/%d max_share=%.2f\n",
                    case,mode,c.method,c.Np,c.K,m.lme,m.se,m.mean,m.sd,m.sec,m.nfinite,reps,m.max_share)
            push!(summary,(case=case,mode=mode,method=c.method,Np=c.Np,K=c.K,budget=budget,
                           reps=reps,exact=isnothing(exact) ? NaN : exact,
                           relative_mean=relative_mean,relative_se=relative_se,z=z,m...,
                           events=mean(x.nevents for x ∈ rows),
                           inner_resamples=mean(x.inner_resamples for x ∈ rows)))
        end
        write_csv(joinpath(RESULTS,"e29_$(case)_summary.csv"),summary)
    end
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
    for (j,c) ∈ enumerate(cases)
        exact = ExactSEIR.loglik(c.g; c.θ...)
        println("$(c.case): exact log likelihood $exact")
        measure_case(c.case,c.g,c.θ; budget=EXACT_BUDGET,reps=EXACT_REPS,exact,
                     modes=("unguided","guided"),seed=291000+100000j)
    end
end

if occursin('B',PARTS)
    for N ∈ POPS
        θ = merge(DEFAULTS,(pop=N,ψ=8/N))
        g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
        println("N=$N: $(g.nsample) samples")
        measure_case("N$N",g,θ; seed=292000+N)
    end
end

if occursin('C',PARTS)
    N = 10000
    θ = merge(DEFAULTS,(pop=N,ψ=8/N))
    g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
    println("Matched-guide follow-up: N=$N, $(g.nsample) samples")
    measure_case("N10000_matched",g,θ; modes=("matched",),seed=303000)
end
