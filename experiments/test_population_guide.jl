using Test, Random, PhyloPOMP
using PhyloPOMP: FastSEIR
using PhyloPOMP.FastSEIR.Demes: Expos, Infec
include("population_guide.jl")
using .PopulationGuide: ForecastTree, pfilter_forecast

@testset "population-aware colour guide" begin
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",time=1.6)
    equilibrium = (β=4.0,σ=1.0,γ=1.0,ω=1.0,ψ=0.02,χ=0.0,
                   pop=100,S0=0.25,E0=0.25,I0=0.25,R0=0.25)
    ft = ForecastTree(g; equilibrium...)
    static = FastSEIR.Tree(g,fsmarkov(Expos=>1,Infec=>1,(Expos,Infec)=>2))
    @test all(==(25.0),ft.population)
    @test ft.intervals[1].toI ≈ static.toI
    @test ft.intervals[1].toE ≈ static.toE
    for k ∈ eachindex(static.time)
        @test ft.intervals[k].present[k] ≈ static.present[k]
    end
    expected = pfilter(static; Np=500,equilibrium...,rng=Xoshiro(30))
    pf = pfilter_forecast(ft; Np=500,rng=Xoshiro(30))
    @test logLik(pf) ≈ logLik(expected)
    @test pf.nevents == expected.nevents

    asymmetric = merge(equilibrium,(σ=2.0,S0=0.25,E0=0.15,I0=0.30,R0=0.30))
    aft = ForecastTree(g; asymmetric...)
    a, b = 2*16/31, 31/16
    ast = FastSEIR.Tree(g,fsmarkov(Expos=>b,Infec=>a,(Expos,Infec)=>a+b))
    @test aft.intervals[1].toI ≈ ast.toI
    @test aft.intervals[1].toE ≈ ast.toE
    for k ∈ eachindex(ast.time)
        @test aft.intervals[k].present[k] ≈ ast.present[k]
    end

    par = (β=4.0,σ=0.8,γ=0.5,ω=0.2,ψ=0.4,χ=0.0,
           pop=7,S0=5/7,E0=0.0,I0=1/7,R0=1/7)
    ft = ForecastTree(g; par...)
    fine = ForecastTree(g; par...,step=0.005)
    @test all(isfinite,ft.population) && all(≥(0),ft.population)
    @test all(x -> isapprox(x,7),sum(ft.population; dims=1))
    @test ft.population ≈ fine.population rtol=1e-6
    @test !all(isapprox(ft.intervals[k].κ,ft.intervals[1].κ) for k ∈ eachindex(ft.intervals))
    pf = pfilter_forecast(ft; Np=500,rng=Xoshiro(31))
    serial = pfilter_forecast(ft; Np=500,rng=Xoshiro(31),threaded=false)
    @test isfinite(logLik(pf))
    @test pf.cond_logLik == serial.cond_logLik
    @test pf.nevents == serial.nevents
    @test sum(cond_logLik(pf)) ≈ logLik(pf)
    @test all(0 .≤ pf.zero_fraction .≤ 1)
    @test_throws AssertionError ForecastTree(g; χ=0.1)
    @test_throws AssertionError ForecastTree(g; step=0)
end
