module ExactSEIRTest

import ..Main: h1, h2

@info h1("exact SEIR likelihood")

using Test
using Random: seed!
using PhyloPOMP
using PhyloPOMP: ExactSEIR, NaiveSEIR, SoftSEIR, GuidedSEIR, HardSEIR
import PartiallyObservedMarkovProcesses as POMP

@testset verbose=true "exact SEIR likelihood" begin

    seed!(1234567)

    @info h2("agreement with an independent mpmath matrix-exponential solver")
    base = (β=2.0, σ=1.5, γ=0.7, ω=0.3, ψ=0.5, χ=0.0)
    cases = [
        ("([&&PhyloPOMP type=sample]:1.1);", 2.0,
         merge(base, (pop=4, S0=0.5, E0=0.0, I0=0.5, R0=0.0)), -2.084108805912760466),
        ("(([&&PhyloPOMP type=sample]:0.6,[&&PhyloPOMP type=sample]:1.1)[&&PhyloPOMP type=node]:0.4);", 2.0,
         merge(base, (pop=4, S0=0.5, E0=0.25, I0=0.25, R0=0.0)), -3.8516558113438508557),
        ("(([&&PhyloPOMP type=sample]:0.7)[&&PhyloPOMP type=sample]:0.5);", 1.6,
         merge(base, (pop=3, S0=1/3, E0=0.0, I0=2/3, R0=0.0)), -2.7365030312415689241),
        (["([&&PhyloPOMP type=sample]:0.8);", "([&&PhyloPOMP type=sample]:1.3);"], 1.5,
         merge(base, (pop=4, S0=0.25, E0=0.25, I0=0.5, R0=0.0)), -2.509430271232934073),
        ("([&&PhyloPOMP type=sample]:1.1);", 2.0,
         merge(base, (χ=0.4, pop=4, S0=0.5, E0=0.0, I0=0.5, R0=0.0)), -2.3709311705480407499),
    ]
    for (nwk, time, par, ref) ∈ cases
        g = parse_newick(nwk; time)
        @test ExactSEIR.loglik(g; par...) ≈ ref atol=1e-12
    end
    g = parse_newick(cases[2][1], time=cases[2][2])
    @test length(ExactSEIR.events(g)) == length(g)

    @info h2("particle filters against the exact likelihood when I == ℓ_I is visited")
    ## a single infective carries the root lineage, so transmissions occur with I == ellI
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    par = (β=4.0, σ=0.7, γ=0.5, ω=0.2, ψ=0.4, χ=0.0, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6)
    exact = ExactSEIR.loglik(g; par...)
    @test exact ≈ -3.1014176307666657 atol=1e-10
    m(M) = fsmarkov(M.Demes.Expos=>0.1, M.Demes.Infec=>1, (M.Demes.Expos,M.Demes.Infec)=>1)
    agrees(p) = begin
        e = POMP.logmeanexp([logLik(pfilter(p, Np=300)) for _ ∈ 1:60], se=true)
        abs(e.est - exact) < 5e.se + 0.02
    end
    @test agrees(GuidedSEIR.filter_pomp(g, m(GuidedSEIR); par...))
    ## kingaa/PhyloPOMP.jl#8: these filters were biased downward before the
    ## fix of the transmission proposal (see test/seir_support.jl).
    @test agrees(NaiveSEIR.filter_pomp(g; par...))
    @test agrees(SoftSEIR.filter_pomp(g, m(SoftSEIR); par...))
    @test agrees(HardSEIR.filter_pomp(g, m(HardSEIR); par...))

end

end
