module SupportTest

import ..Main: h1, h2

@info h1("support of the transmission proposal")

using Test
using Random: seed!
using PhyloPOMP
using PhyloPOMP: NaiveSEIR, SoftSEIR, GuidedSEIR, HardSEIR
import PartiallyObservedMarkovProcesses as POMP

## Regression test for kingaa/PhyloPOMP.jl#8: when every infective carries a
## lineage (I == ellI), a transmission can still leave the coloring unchanged,
## so the proposal must give that outcome positive probability.

@testset verbose=true "transmission proposal support" begin

    seed!(20261028)

    ## a single infective carries the root lineage, so transmissions occur with I == ellI
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    par = (β=4.0, σ=0.7, γ=0.5, ω=0.2, ψ=0.4, χ=0.0, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6)
    ## exact log likelihood, from a direct (uniformization) solution of the filter
    ## equation, cross-checked against an independent matrix-exponential solver
    exact = -3.1014176307666657

    m(M) = fsmarkov(M.Demes.Expos=>0.1, M.Demes.Infec=>1, (M.Demes.Expos,M.Demes.Infec)=>1)
    est(p) = POMP.logmeanexp([logLik(pfilter(p, Np=300)) for _ ∈ 1:60], se=true)

    for (name, p) ∈ (
        "naive" => NaiveSEIR.filter_pomp(g; par...),
        "soft" => SoftSEIR.filter_pomp(g, m(SoftSEIR); par...),
        "hard" => HardSEIR.filter_pomp(g, m(HardSEIR); par...),
        "guided" => GuidedSEIR.filter_pomp(g, m(GuidedSEIR); par...),
    )
        e = est(p)
        @info "$name: $(round(e.est, digits=3)) ± $(round(e.se, digits=3)) (exact $(round(exact, digits=3)))"
        @test abs(e.est - exact) < 5e.se + 0.02
    end

end

end
