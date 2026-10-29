using Test, Random, PhyloPOMP
using PhyloPOMP: FastSEIR
include("nested_colour_filter.jl")
using .NestedColour
const NC = NestedColour

@testset "nested colour filter" begin
    tr = FastSEIR.Tree(parse_newick("([&&PhyloPOMP type=sample]:1.0);",time=2.0))
    p = (β=0.0,σ=0.0,γ=100.0,ω=0.0,ψ=0.0,pop=1.0)
    for labelled ∈ (false,true)
        P = FastSEIR.Particles(1,tr.nwords)
        C = FastSEIR.Particles(1,tr.nwords)
        P.I[1] = 1
        P.ellI[1] = Int(labelled)
        labelled && FastSEIR.infec!(P,1,tr.lins[tr.ptr[1]])
        ll, n, _ = NC.regular_group!(tr,P,C,1,labelled ? 1 : 2,p,NC.Workspace(1),
                                     Xoshiro(29),false,0.5,1)
        @test n == 1
        if labelled
            @test ll == -Inf
        else
            @test ll == 0.0
            @test P.I[1] == 0 && P.R[1] == 1
        end
    end

    P = FastSEIR.Particles(2,1); C = FastSEIR.Particles(2,1)
    P.ll .= log.([2.0,6.0])
    w = NC.Workspace(2)
    @test NC.settle!(P,C,1,w,Xoshiro(1),0.0) == (0.0,0)
    gain, nr = NC.settle!(P,C,1,w,Xoshiro(1),0.0; force=true)
    @test gain ≈ log(4)
    @test nr == 1 && all(iszero,P.ll)
    P.ll .= -Inf
    @test NC.settle!(P,C,1,w,Xoshiro(1),0.5) == (-Inf,0)

    g = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    tr = FastSEIR.Tree(g)
    par = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0,
           pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7)
    for K ∈ (1,4), check ∈ (1,32)
        pf = NC.pfilter_nested(tr; Np=200,K,inner_check=check,par...,rng=Xoshiro(29))
        serial = NC.pfilter_nested(tr; Np=200,K,inner_check=check,par...,rng=Xoshiro(29),threaded=false)
        @test isfinite(logLik(pf))
        @test pf.cond_logLik == serial.cond_logLik
        @test pf.nevents == serial.nevents
        @test pf.inner_resamples == serial.inner_resamples
        @test sum(cond_logLik(pf)) ≈ logLik(pf)
        @test all(1 .≤ eff_sample_size(pf) .≤ 200+1e-10)
    end
    @test_throws AssertionError NC.pfilter_nested(tr; χ=0.1)
    @test_throws AssertionError NC.pfilter_nested(tr; K=0)
    @test_throws AssertionError NC.pfilter_nested(tr; inner_check=0)
end
