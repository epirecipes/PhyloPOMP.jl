## E32: freeze an independently learned interval particle budget.
include("common.jl")
include("particle_allocation.jl")
using .ParticleAllocation: pfilter_allocated, allocate

const PARTS = get(ENV,"E32_PARTS","ABC")
const EXACT_REPS = parse(Int,get(ENV,"E32_EXACT_REPS","64"))
const EXACT_NP = parse(Int,get(ENV,"E32_EXACT_NP","1000"))
const REPS = parse(Int,get(ENV,"E32_REPS","32"))
const NP = parse(Int,get(ENV,"E32_NP","1000"))
const LARGE_NP = parse(Int,get(ENV,"E32_LARGE_NP","512"))
const PILOT_REPS = parse(Int,get(ENV,"E32_PILOT_REPS","8"))
const PILOT_NP = parse(Int,get(ENV,"E32_PILOT_NP","256"))
@assert min(EXACT_REPS,REPS,PILOT_REPS) ≥ 2
@assert min(EXACT_NP,NP,LARGE_NP,PILOT_NP) ≥ 16

function learn_schedule(case, tr, θ, uniform, seed)
    M = length(tr.time)
    k0 = findlast(==(tr.time[1]),tr.time)
    fixed = fill(PILOT_NP,M)
    pfilter_allocated(tr,fixed; θ...,rng=Xoshiro(seed))
    pilots = NamedTuple[]
    variances = zeros(M)
    costs = zeros(M)
    seconds = 0.0
    for r ∈ 1:PILOT_REPS
        result = @timed pfilter_allocated(tr,fixed; θ...,rng=Xoshiro(seed+r))
        pf, events = result.value
        @assert isfinite(logLik(pf)) "pilot collapsed; cannot learn a complete schedule"
        seconds += result.time
        for k ∈ k0:M
            a = PILOT_NP*max(PILOT_NP/pf.eff_sample_size[k]-1,0.0)
            c = events[k]/PILOT_NP+tr.nwords+1+(k == k0 ? k0 : 0)
            variances[k] += a/PILOT_REPS
            costs[k] += c/PILOT_REPS
            push!(pilots,(case=case,replicate=r,seed=seed+r,k=k,
                          Np=PILOT_NP,loglik=logLik(pf),replicate_sec=result.time,
                          increment=pf.cond_logLik[k],ess=pf.eff_sample_size[k],
                          nevents=events[k],variance_proxy=a,work_per_particle=c))
        end
    end
    quantum = 16
    lower = max(quantum,ceil(Int,uniform/8/quantum)*quantum)
    upper = floor(Int,8uniform/quantum)*quantum
    plan = @timed allocate(variances[k0:end],costs[k0:end],uniform;
                           lower,upper,quantum)
    sizes = [fill(plan.value[1],k0-1); plan.value]
    schedule = [(case=case,k=k,time=tr.time[k],tend=tr.tend[k],active=k ≥ k0,
                 uniform_N=uniform,allocated_N=sizes[k],variance_proxy=variances[k],
                 work_per_particle=costs[k]) for k ∈ 1:M]
    write_csv(joinpath(RESULTS,"e32_$(case)_pilot.csv"),pilots)
    write_csv(joinpath(RESULTS,"e32_$(case)_schedule.csv"),schedule)
    (sizes=sizes,seconds=seconds+plan.time,
     work_ratio=sum(sizes .* costs)/(uniform*sum(costs)),k0=k0)
end

function run_filter(tr, θ, method, sizes, seed)
    if method == "canonical"
        pfilter(tr; Np=sizes[1],θ...,rng=Xoshiro(seed))
    elseif method ∈ ("uniform","allocated","alternating")
        pfilter_allocated(tr,sizes; θ...,rng=Xoshiro(seed)).pf
    else
        error("unknown method $method")
    end
end

function metrics(rows)
    ll = [r.loglik for r ∈ rows]
    nfinite = count(isfinite,ll)
    sec = mean(r.sec for r ∈ rows)
    if nfinite == 0
        return (lme=-Inf,se=Inf,mean=-Inf,sd=Inf,cv2=Inf,variance_cost=Inf,
                max_share=NaN,nfinite,sec)
    end
    a = exp.(ll .- maximum(ll))
    cv2 = var(a)/mean(a)^2
    (lme=maximum(ll)+log(mean(a)),se=sqrt(cv2/length(ll)),mean=mean(ll),
     sd=nfinite == length(ll) ? std(ll) : Inf,cv2,variance_cost=cv2*sec,
     max_share=maximum(a)/sum(a),nfinite,sec)
end

function measure_case(case, g, θ; Np=NP, reps=REPS, guided=true,
                      exact=nothing, seed=320000)
    tr = guided ? FastSEIR.Tree(g,default_guide()) : FastSEIR.Tree(g)
    M = length(tr.time)
    fixed = fill(Np,M)
    k0 = findlast(==(tr.time[1]),tr.time)
    plan = if isnothing(exact)
        learn_schedule(case,tr,θ,Np,seed+10000000)
    else
        (sizes=[isodd(k) ? Np÷4 : 2Np for k ∈ 1:M],
         seconds=0.0,work_ratio=NaN,k0=k0)
    end
    configs = ((method="canonical",sizes=fixed),(method="uniform",sizes=fixed),
               (method=isnothing(exact) ? "allocated" : "alternating",sizes=plan.sizes))
    for c ∈ configs
        run_filter(tr,θ,c.method,c.sizes,seed)
    end
    raw = NamedTuple[]
    summary = NamedTuple[]
    for r ∈ 1:reps
        for i ∈ shuffle(Xoshiro(seed+r),collect(eachindex(configs)))
            c = configs[i]
            run_seed = seed+10000+r
            result = @timed run_filter(tr,θ,c.method,c.sizes,run_seed)
            pf = result.value
            @assert isfinite(logLik(pf)) || logLik(pf) == -Inf "invalid likelihood"
            push!(raw,(case=case,method=c.method,Np=Np,replicate=r,seed=run_seed,
                       loglik=logLik(pf),sec=result.time,bytes=result.bytes,
                       nevents=pf.nevents))
        end
        write_csv(joinpath(RESULTS,"e32_$(case)_raw.csv"),raw)
        (r % 8 == 0 || r == reps) && println("$case: replicate $r/$reps")
    end
    for c ∈ configs
        rows = filter(x -> x.method == c.method,raw)
        m = metrics(rows)
        relative_mean, relative_se, z = if isnothing(exact)
            (NaN,NaN,NaN)
        else
            a = exp.([x.loglik-exact for x ∈ rows])
            avg, se = mean(a), std(a)/sqrt(reps)
            @assert abs(avg-1) < 5se+0.01 "$case $(c.method) disagrees with exact"
            (avg,se,(avg-1)/se)
        end
        pilot_sec = c.method == "allocated" ? plan.seconds : 0.0
        @printf("%-15s %-11s N=%4d range=%d:%d SD=%5.2f %.3fs lme=%.2f +/- %.2f finite=%d/%d share=%.2f\n",
                case,c.method,Np,minimum(c.sizes[k0:end]),maximum(c.sizes[k0:end]),
                m.sd,m.sec,m.lme,m.se,m.nfinite,reps,m.max_share)
        push!(summary,(case=case,method=c.method,Np=Np,reps=reps,guided,
                       exact=isnothing(exact) ? NaN : exact,relative_mean,relative_se,z,
                       m...,pilot_sec,amortized_sec=m.sec+pilot_sec/reps,
                       planned_work_ratio=c.method == "allocated" ? plan.work_ratio :
                           c.method == "alternating" ? NaN : 1.0,
                       minimum_N=minimum(c.sizes[k0:end]),maximum_N=maximum(c.sizes[k0:end]),
                       mean_N=mean(c.sizes[k0:end]),mean_events=mean(r.nevents for r ∈ rows)))
    end
    write_csv(joinpath(RESULTS,"e32_$(case)_summary.csv"),summary)
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
    for (i,c) ∈ enumerate(cases), (j,guided) ∈ enumerate((false,true))
        exact = ExactSEIR.loglik(c.g; c.θ...)
        measure_case("$(c.case)_$(guided ? "guided" : "none")",c.g,c.θ;
                     Np=EXACT_NP,reps=EXACT_REPS,guided,exact,seed=321000+100000i+1000j)
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
    measure_case("E19_fit",g,θ; seed=322000)
    g2 = parse_newick(FastSEIR.seir_trees[2])
    θ2 = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    measure_case("tree2_fit",g2,θ2; seed=323000)
end

if occursin('C',PARTS)
    for N ∈ (1000,10000)
        local θ = merge(DEFAULTS,(pop=N,ψ=8/N))
        local g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
        measure_case("N$N",g,θ; Np=LARGE_NP,seed=324000+N)
    end
end
