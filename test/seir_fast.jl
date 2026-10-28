module FastSEIRTest

import ..Main: h1, h2

@info h1("fast streaming SEIR filter")

using Test
using Random: Xoshiro, seed!
using PhyloPOMP
using PhyloPOMP.FastSEIR
using PhyloPOMP.FastSEIR.Demes: Expos, Infec

@testset verbose=true "fast streaming SEIR filter" begin

    seed!(4242)
    m = fsmarkov(Expos=>0.1, Infec=>1, (Expos,Infec)=>1)

    @info h2("agreement with the exact likelihood")
    ## a single infective carries the root lineage, so transmissions occur with I == ellI
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    par = (β=4.0, σ=0.7, γ=0.5, ω=0.2, ψ=0.4, χ=0.0, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6)
    ## exact log likelihood (direct uniformization solution of the filter equation,
    ## cross-checked against an independent matrix-exponential solver)
    exact = -3.1014176307666657
    est(tr; kwargs...) = logmeanexp([logLik(pfilter(tr; Np=300, par..., kwargs...)) for _ ∈ 1:60], se=true)
    tr = FastSEIR.Tree(g, m)
    for e ∈ (
        est(tr),
        est(tr, proposal=:uniform),
        est(tr, guided=false),
        est(tr, trigger=0.5),
        est(FastSEIR.Tree(g)),
    )
        @test abs(e.est - exact) < 5e.se + 0.02
    end

    @info h2("diagnostics and reproducibility")
    g = parse_newick(FastSEIR.seir_trees[1], time=50.0)
    tr = FastSEIR.Tree(g, m)
    pf = pfilter(tr, Np=500, rng=Xoshiro(17))
    @test pf isa FastSEIR.FastPfilterd
    @test isfinite(logLik(pf))
    @test length(eff_sample_size(pf)) == length(cond_logLik(pf)) == length(g)
    @test sum(cond_logLik(pf)) ≈ logLik(pf)
    @test all(0 .≤ pf.zero_fraction .≤ 1)
    @test pf.nevents > 0
    ## the result depends on the random-number generator, not on the threads
    @test logLik(pf) == logLik(pfilter(tr, Np=500, rng=Xoshiro(17), threaded=false))

end

end
