module ExactSEIRTest

import ..Main: h1, h2

@info h1("exact SEIR likelihood and filter validation")

using Test
using Random: seed!
using PhyloPOMP
using PhyloPOMP: ExactSEIR, SimSEIR, FastSEIR, NaiveSEIR, SoftSEIR, GuidedSEIR, HardSEIR
import PartiallyObservedMarkovProcesses as POMP

lme_se(ll) = begin
    x = POMP.logmeanexp(ll, se=true)
    x.est, x.se
end

@testset verbose=true "exact SEIR likelihood" begin

    seed!(1234567)

    @info h2("reference value (independent mpmath solver: -2.084108805912760466)")
    g = parse_newick("([&&PhyloPOMP type=sample]:1.1);", time=2.0)
    par = (β=2.0, σ=1.5, γ=0.7, ω=0.3, ψ=0.5, χ=0.0, pop=4, S0=0.5, E0=0.0, I0=0.5, R0=0.0)
    @test ExactSEIR.loglik(g; par...) ≈ -2.084108805912760466 atol=1e-12

    @info h2("simulator")
    sim = SimSEIR.simulate(; par..., time=2.0)
    @test sim.genealogy isa Genealogy
    @test sim.nsample == sim.genealogy.nsample

    @info h2("all filters agree with the exact likelihood when I == ℓ_I is visited")
    ## a single infective carries the root lineage, so transmissions occur with I == ℓ_I
    g = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    par = (β=4.0, σ=0.7, γ=0.5, ω=0.2, ψ=0.4, χ=0.0, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6)
    exact = ExactSEIR.loglik(g; par...)
    R, Np = 60, 300
    pomps = (
        naive = NaiveSEIR.filter_pomp(g; par...),
        soft = SoftSEIR.filter_pomp(g, fsmarkov(SoftSEIR.Demes.Expos=>0.1, SoftSEIR.Demes.Infec=>1,
            (SoftSEIR.Demes.Expos, SoftSEIR.Demes.Infec)=>1); par...),
        guided = GuidedSEIR.filter_pomp(g, fsmarkov(GuidedSEIR.Demes.Expos=>0.1, GuidedSEIR.Demes.Infec=>1,
            (GuidedSEIR.Demes.Expos, GuidedSEIR.Demes.Infec)=>1); par...),
        hard = HardSEIR.filter_pomp(g, fsmarkov(HardSEIR.Demes.Expos=>0.1, HardSEIR.Demes.Infec=>1,
            (HardSEIR.Demes.Expos, HardSEIR.Demes.Infec)=>1); par...),
    )
    for (name, p) ∈ pairs(pomps)
        est, se = lme_se([logLik(pfilter(p, Np=Np)) for _ ∈ 1:R])
        @info "$name: $(round(est, digits=3)) ± $(round(se, digits=3)) (exact $(round(exact, digits=3)))"
        @test abs(est - exact) < 5se + 0.02
    end
    m = fsmarkov(GuidedSEIR.Demes.Expos=>0.1, GuidedSEIR.Demes.Infec=>1,
                 (GuidedSEIR.Demes.Expos, GuidedSEIR.Demes.Infec)=>1)
    tr = FastSEIR.Tree(g, guide(g, m, GuidedSEIR.knowledge!))
    for opt ∈ (FastSEIR.Options(), FastSEIR.Options(proposal=:fixed, order=:pre),
               FastSEIR.Options(lookahead=:both))
        est, se = lme_se(FastSEIR.logliks(tr, par; Np, n=R, opt))
        @test abs(est - exact) < 5se + 0.02
    end
    ## the naive proposal without the support fix is biased: keep it as a canary
    est, _ = lme_se(FastSEIR.logliks(tr, par; Np, n=R,
                    opt=FastSEIR.Options(proposal=:naive, guided=false, order=:pre)))
    @test est < exact - 1

end

end
