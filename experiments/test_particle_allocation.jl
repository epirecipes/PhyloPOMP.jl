using Test, Random, Statistics, PhyloPOMP
using PhyloPOMP: FastSEIR
include("particle_allocation.jl")
using .ParticleAllocation: pfilter_allocated, allocate, resample!

@testset "independent-pilot particle allocation" begin
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    tr = FastSEIR.Tree(g)
    θ = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0,
         pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7)
    sizes = fill(500,length(tr.time))
    expected = pfilter(tr; Np=500,θ...,rng=Xoshiro(32))
    actual = pfilter_allocated(tr,sizes; θ...,rng=Xoshiro(32))
    @test actual.pf.cond_logLik == expected.cond_logLik
    @test actual.pf.eff_sample_size == expected.eff_sample_size
    @test actual.pf.nevents == expected.nevents == sum(actual.events)
    sizes = [isodd(k) ? 80 : 300 for k ∈ eachindex(tr.time)]
    parallel = pfilter_allocated(tr,sizes; θ...,rng=Xoshiro(33))
    serial = pfilter_allocated(tr,sizes; θ...,rng=Xoshiro(33),threaded=false)
    @test parallel.pf.cond_logLik == serial.pf.cond_logLik
    @test parallel.events == serial.events
    @test logLik(parallel.pf) ≈ sum(cond_logLik(parallel.pf))
    @test all(parallel.pf.eff_sample_size .≤ sizes .* (1+1e-12))
    @test_throws AssertionError pfilter_allocated(tr,[0; sizes[2:end]]; θ...)
    @test_throws AssertionError pfilter_allocated(tr,sizes[2:end]; θ...)

    counts = zeros(3)
    ancestors = zeros(Int,7)
    rng = Xoshiro(34)
    for _ ∈ 1:2000
        resample!(ancestors,[0.1,0.2,0.7],rng)
        for a ∈ ancestors
            counts[a] += 1
        end
    end
    @test counts/2000 ≈ 7 .* [0.1,0.2,0.7] atol=0.03
    @test allocate([1.0,16.0],[1.0,1.0],100; lower=25,upper=400,quantum=1) == [40,160]
    n = allocate([1.0,1.0],[1.0,4.0],100; lower=25,upper=400,quantum=1)
    @test n[1] > n[2]
    @test abs(sum(n .* [1,4])-500) ≤ 3
    @test_throws AssertionError allocate(zeros(2),ones(2),100; lower=16,upper=400)
    @test_throws AssertionError allocate(ones(2),[0.0,1.0],100; lower=16,upper=400)
end
