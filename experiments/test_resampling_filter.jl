using Test, Random, Statistics, PhyloPOMP
using PhyloPOMP: FastSEIR
include("resampling_filter.jl")
using .ResamplingFilter: pfilter_resampling, resample!

@testset "alternative ancestor resampling" begin
    for method ∈ (:systematic,:stratified,:residual)
        a = zeros(Int,4)
        scratch = zeros(4)
        resample!(a,fill(0.25,4),scratch,method,Xoshiro(330))
        @test a == 1:4
        resample!(a,[0.0,0.0,1.0,0.0],scratch,method,Xoshiro(331))
        @test all(==(3),a)
        w = [0.1,0.05,0.0,0.2,0.15,0.25,0.25]
        a, scratch = zeros(Int,7), zeros(7)
        counts = zeros(7)
        rng = Xoshiro(332)
        for _ ∈ 1:4000
            resample!(a,w,scratch,method,rng)
            for j ∈ a
                counts[j] += 1
            end
        end
        @test counts/4000 ≈ 7 .* w atol=0.04
        if method == :residual
            offspring = [count(==(j),a) for j ∈ 1:7]
            @test all(offspring .≥ floor.(Int,7 .* w))
        end
    end

    g = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    tr = FastSEIR.Tree(g)
    θ = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0,
         pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7)
    expected = pfilter(tr; Np=500,θ...,rng=Xoshiro(333))
    actual = pfilter_resampling(tr; Np=500,θ...,rng=Xoshiro(333))
    @test actual.cond_logLik == expected.cond_logLik
    @test actual.eff_sample_size == expected.eff_sample_size
    @test actual.nevents == expected.nevents
    for method ∈ (:stratified,:residual)
        parallel = pfilter_resampling(tr; Np=500,θ...,resampling=method,rng=Xoshiro(334))
        serial = pfilter_resampling(tr; Np=500,θ...,resampling=method,rng=Xoshiro(334),threaded=false)
        @test parallel.cond_logLik == serial.cond_logLik
        @test parallel.nevents == serial.nevents
        @test logLik(parallel) ≈ sum(cond_logLik(parallel))
    end
    @test_throws AssertionError pfilter_resampling(tr; Np=0)
    @test_throws AssertionError pfilter_resampling(tr; resampling=:unknown)
end
