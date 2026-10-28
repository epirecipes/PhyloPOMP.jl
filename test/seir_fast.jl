module FastSEIRTest

import ..Main: h1, h2

@info h1("fast streaming SEIR filter")

using Test
using Random: Xoshiro, seed!
using PhyloPOMP
using PhyloPOMP: FastSEIR, GuidedSEIR
import PartiallyObservedMarkovProcesses as POMP

@testset verbose=true "fast streaming SEIR filter" begin

    seed!(4242)
    m = fsmarkov(GuidedSEIR.Demes.Expos=>0.1, GuidedSEIR.Demes.Infec=>1,
                 (GuidedSEIR.Demes.Expos, GuidedSEIR.Demes.Infec)=>1)
    mktree(g) = FastSEIR.Tree(g, guide(g, m, GuidedSEIR.knowledge!))

    @info h2("agreement with the exact likelihood")
    ## a single infective carries the root lineage, so transmissions occur with I == ellI
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    par = (β=4.0, σ=0.7, γ=0.5, ω=0.2, ψ=0.4, χ=0.0, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6)
    ## exact log likelihood (direct uniformization solution of the filter equation,
    ## cross-checked against an independent matrix-exponential solver)
    exact = -3.1014176307666657
    tr = mktree(g)
    est(opt) = POMP.logmeanexp(FastSEIR.logliks(tr, par; Np=300, n=60, opt), se=true)
    for opt ∈ (
        FastSEIR.Options(),                                         # local/post/guided
        FastSEIR.Options(proposal=:fixed, order=:pre),
        FastSEIR.Options(proposal=:local, guided=false),
        FastSEIR.Options(resample=:ess, trigger=0.5),
        FastSEIR.Options(lookahead=:both, horizon=2),
    )
        e = est(opt)
        @test abs(e.est - exact) < 5e.se + 0.02
    end
    ## the paper's naive proposal violates the support condition (kingaa/PhyloPOMP.jl#8)
    e = est(FastSEIR.Options(proposal=:naive, guided=false, order=:pre))
    @test e.est < exact - 1

    @info h2("reproducibility, threading and the kernel hook")
    g1 = parse_newick(GuidedSEIR.seir_trees[1], time=50.0)
    par1 = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
    tr1 = mktree(g1)
    run(opt) = FastSEIR.pfilter(tr1, par1; Np=500, opt, rng=Xoshiro(17)).logLik
    l1 = run(FastSEIR.Options(threaded=true))
    @test isfinite(l1)
    @test l1 == run(FastSEIR.Options(threaded=false))
    hook = (tr, P, i, k, t, tf, p, opt, rng) -> FastSEIR.regular!(tr, P, i, k, t, tf, p, opt, rng)
    @test l1 == run(FastSEIR.Options(regular=hook))

    @info h2("diagnostics")
    res = FastSEIR.pfilter(tr1, par1; Np=200, diagnostics=true)
    @test length(res.ess) == length(res.zerofrac) == length(res.cll) == length(g1)
    @test res.nevents > 0
    @test all(x -> isnan(x) || 0 ≤ x ≤ 1, res.zerofrac)

end

end
