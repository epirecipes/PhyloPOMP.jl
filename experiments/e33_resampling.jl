## E33: isolate the variance and cost of selecting particle ancestors.
include("common.jl")
include("resampling_filter.jl")
using .ResamplingFilter: pfilter_resampling

const PARTS = get(ENV,"E33_PARTS","ABCD")
const EXACT_REPS = parse(Int,get(ENV,"E33_EXACT_REPS","64"))
const EXACT_NP = parse(Int,get(ENV,"E33_EXACT_NP","1000"))
const REPS = parse(Int,get(ENV,"E33_REPS","32"))
const NP = parse(Int,get(ENV,"E33_NP","1000"))
const TREE2_NP = parse(Int,get(ENV,"E33_TREE2_NP","4000"))
const LARGE_NP = parse(Int,get(ENV,"E33_LARGE_NP","512"))
@assert min(EXACT_REPS,REPS) ≥ 2 && min(EXACT_NP,NP,TREE2_NP,LARGE_NP) > 0

run_filter(tr, θ, method, Np, seed) = if method == "canonical"
    pfilter(tr; Np,θ...,rng=Xoshiro(seed))
elseif method ∈ ("systematic","stratified","residual")
    pfilter_resampling(tr; Np,θ...,resampling=Symbol(method),rng=Xoshiro(seed))
else
    error("unknown method $method")
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

function efficiency(rows, baseline, seed)
    @assert [r.replicate for r ∈ rows] == [r.replicate for r ∈ baseline]
    if !all(r -> isfinite(r.loglik),[rows; baseline])
        return (efficiency_ratio=NaN,bootstrap_low=NaN,bootstrap_high=NaN)
    end
    x = exp.([r.loglik for r ∈ rows] .- maximum(r.loglik for r ∈ rows))
    y = exp.([r.loglik for r ∈ baseline] .- maximum(r.loglik for r ∈ baseline))
    tx, ty = [r.sec for r ∈ rows], [r.sec for r ∈ baseline]
    cost(v,t) = var(v)/mean(v)^2*mean(t)
    ratio = cost(y,ty)/cost(x,tx)
    rng = Xoshiro(seed)
    boot = [begin
        i = rand(rng,1:length(rows),length(rows))
        cost(y[i],ty[i])/cost(x[i],tx[i])
    end for _ ∈ 1:3000]
    (efficiency_ratio=ratio,bootstrap_low=quantile(boot,0.025),
     bootstrap_high=quantile(boot,0.975))
end

function measure_case(case, g, θ; Np=NP, reps=REPS, guided=true,
                      exact=nothing, seed=330000,
                      methods=("canonical","systematic","stratified","residual"))
    tr = guided ? FastSEIR.Tree(g,default_guide()) : FastSEIR.Tree(g)
    @assert "canonical" ∈ methods
    for method ∈ methods
        run_filter(tr,θ,method,Np,seed)
    end
    raw = NamedTuple[]
    summary = NamedTuple[]
    for r ∈ 1:reps
        for i ∈ shuffle(Xoshiro(seed+r),collect(eachindex(methods)))
            run_seed = seed+10000+r
            result = @timed run_filter(tr,θ,methods[i],Np,run_seed)
            pf = result.value
            @assert isfinite(logLik(pf)) || logLik(pf) == -Inf "invalid likelihood"
            push!(raw,(case=case,method=methods[i],Np=Np,replicate=r,seed=run_seed,
                       loglik=logLik(pf),sec=result.time,bytes=result.bytes,
                       nevents=pf.nevents))
        end
        write_csv(joinpath(RESULTS,"e33_$(case)_raw.csv"),raw)
        (r % 8 == 0 || r == reps) && println("$case: replicate $r/$reps")
    end
    baseline = filter(x -> x.method == "canonical",raw)
    if "systematic" ∈ methods
        control = filter(x -> x.method == "systematic",raw)
        @assert [r.loglik for r ∈ baseline] == [r.loglik for r ∈ control]
        @assert [r.nevents for r ∈ baseline] == [r.nevents for r ∈ control]
    end
    for (i,method) ∈ enumerate(methods)
        rows = filter(x -> x.method == method,raw)
        m = metrics(rows)
        relative_mean, relative_se, z = if isnothing(exact)
            (NaN,NaN,NaN)
        else
            a = exp.([x.loglik-exact for x ∈ rows])
            avg, se = mean(a), std(a)/sqrt(reps)
            @assert abs(avg-1) < 5se+0.01 "$case $method disagrees with exact"
            (avg,se,(avg-1)/se)
        end
        e = isnothing(exact) ? efficiency(rows,baseline,seed+20000+i) :
            (efficiency_ratio=NaN,bootstrap_low=NaN,bootstrap_high=NaN)
        @printf("%-15s %-10s N=%4d SD=%5.2f %.3fs lme=%.2f +/- %.2f finite=%d/%d share=%.2f efficiency=%.2f [%.2f,%.2f]\n",
                case,method,Np,m.sd,m.sec,m.lme,m.se,m.nfinite,reps,m.max_share,
                e.efficiency_ratio,e.bootstrap_low,e.bootstrap_high)
        push!(summary,(case=case,method=method,Np=Np,reps=reps,guided,
                       exact=isnothing(exact) ? NaN : exact,relative_mean,relative_se,z,
                       m...,e...,mean_events=mean(r.nevents for r ∈ rows)))
    end
    write_csv(joinpath(RESULTS,"e33_$(case)_summary.csv"),summary)
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
                     Np=EXACT_NP,reps=EXACT_REPS,guided,exact,seed=331000+100000i+1000j)
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
    measure_case("E19_fit",g,θ; seed=332000)
    g2 = parse_newick(FastSEIR.seir_trees[2])
    θ2 = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    measure_case("tree2_fit",g2,θ2; Np=TREE2_NP,seed=333000)
end

if occursin('C',PARTS)
    for N ∈ (1000,10000)
        local θ = merge(DEFAULTS,(pop=N,ψ=8/N))
        local g = SimSEIR.simulate(; θ...,time=40.0,rng=Xoshiro(27)).genealogy
        measure_case("N$N",g,θ; Np=LARGE_NP,seed=334000+N)
    end
end

if occursin('D',PARTS)
    g = parse_newick(FastSEIR.seir_trees[2])
    θ = merge(DEFAULTS,(β=2.806,σ=0.599,γ=0.478,ω=5.171,ψ=0.0271,pop=173))
    measure_case("tree2_confirm",g,θ; Np=TREE2_NP,seed=337000,
                 methods=("canonical","residual"))
end
