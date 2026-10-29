using Test, Random, Statistics, PhyloPOMP
using PhyloPOMP: FastSEIR, ExactSEIR
include("alive_filter.jl")
using .AliveFilter: pfilter_alive, TrialLimit

@testset "weighted alive filter" begin
    g = parse_newick("([&&PhyloPOMP type=sample]:1.0);",time=1.0)
    tr = FastSEIR.Tree(g)
    par = (β=0.0,σ=0.0,γ=0.0,ω=0.0,ψ=0.4,χ=0.0,
           pop=2,S0=0.5,E0=0.0,I0=0.5,R0=0.0)
    exact = ExactSEIR.loglik(g; par...)
    for alive ∈ (false,true)
        result = pfilter_alive(tr; Np=7,alive,par...,rng=Xoshiro(31))
        @test logLik(result.pf) ≈ exact
        @test all(==(alive ? 8 : 7),result.trials)
        @test logLik(result.pf) ≈ sum(cond_logLik(result.pf))
    end

    ## Root proposals choose E or I equally; only I survives. Positive trial
    ## weights include the root proposal correction, sampling and survival.
    rare = merge(par,(S0=0.0,E0=0.5))
    exact_rare = ExactSEIR.loglik(g; rare...)
    values = [exp(logLik(pfilter_alive(tr; Np=3,rare...,threaded=false,
                                      rng=Xoshiro(31000+r)).pf)-exact_rare) for r ∈ 1:1500]
    @test abs(mean(values)-1) < 5std(values)/sqrt(length(values))

    g2 = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    tr2 = FastSEIR.Tree(g2)
    variable = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0,
                pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7)
    threaded = pfilter_alive(tr2; Np=100,variable...,rng=Xoshiro(32))
    serial = pfilter_alive(tr2; Np=100,variable...,rng=Xoshiro(32),threaded=false)
    @test threaded.pf.cond_logLik == serial.pf.cond_logLik
    @test threaded.trials == serial.trials
    @test threaded.proposed == serial.proposed
    @test threaded.pf.nevents == serial.pf.nevents
    @test all(threaded.trials .≥ 101)
    @test all(threaded.proposed .≥ threaded.trials)
    @test all(iszero,threaded.pf.zero_fraction)

    impossible = merge(par,(S0=0.5,E0=0.5,I0=0.0))
    err = try
        pfilter_alive(tr; Np=2,max_trials=5,impossible...,rng=Xoshiro(33))
        nothing
    catch e
        e isa TrialLimit || rethrow()
        e
    end
    @test err isa TrialLimit
    @test err.trials == 5
    @test err.successes == 0
    ordinary = pfilter_alive(tr; Np=2,alive=false,impossible...,rng=Xoshiro(33))
    @test logLik(ordinary.pf) == -Inf
    @test_throws AssertionError pfilter_alive(tr; Np=0)
    @test_throws AssertionError pfilter_alive(tr; Np=2,max_trials=2)
end
