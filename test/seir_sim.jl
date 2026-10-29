module SimSEIRTest

import ..Main: h1, h2

@info h1("SEIR genealogy simulator")

using Test
using Random: Xoshiro, seed!
using PhyloPOMP
using PhyloPOMP: SimSEIR, GuidedSEIR, Root, Node, Sample
import PartiallyObservedMarkovProcesses as POMP

@testset verbose=true "SEIR genealogy simulator" begin

    seed!(1729)
    par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)

    @info h2("structure")
    sim = SimSEIR.simulate(; par..., time=50.0, rng=Xoshiro(1))
    g = sim.genealogy
    @test g isa Genealogy{PhyloPOMP.Unstructured}
    @test sim.nsample == g.nsample
    @test g.nsample > 0
    @test timezero(g) == 0.0 && g.time == 50.0
    @test issorted([n.slate for n ∈ g.nodes])
    @test all(g.nodes) do n
        (n.type == Root && length(n.children) == 1) ||
            (n.type == Sample && length(n.children) < 2) ||
            (n.type == Node && length(n.children) == 2)
    end
    @test count(n -> n.type == Sample, g.nodes) == g.nsample
    @test sum(values(sim.state)) == par.pop      # no destructive sampling

    @info h2("reproducibility and Newick round trip")
    g2 = SimSEIR.simulate(; par..., time=50.0, rng=Xoshiro(1)).genealogy
    @test newick(g) == newick(g2)
    h = parse_newick(newick(g), time=g.time)
    @test length(h) == length(g) && h.nsample == g.nsample

    @info h2("destructive sampling removes infectives")
    d = SimSEIR.simulate(; par..., ψ=0.0, χ=0.05, time=50.0, rng=Xoshiro(3))
    @test sum(values(d.state)) == par.pop - d.nsample

    @info h2("filters accept simulated genealogies")
    m = fsmarkov(GuidedSEIR.Demes.Expos=>0.1, GuidedSEIR.Demes.Infec=>1,
                 (GuidedSEIR.Demes.Expos, GuidedSEIR.Demes.Infec)=>1)
    @test isfinite(logLik(pfilter(GuidedSEIR.filter_pomp(g, m; par...), Np=200)))

    @info h2("curtailing and thinning genealogies")
    @test curtail(g, g.time) == g
    @test thin(g, 1) == g
    @test thin(g, 0).nsample == 0
    ## the history up to a time does not depend on how long the simulation runs
    for seed ∈ 1:5
        whole = SimSEIR.simulate(; par..., time=50.0, rng=Xoshiro(seed)).genealogy
        part = SimSEIR.simulate(; par..., time=30.0, rng=Xoshiro(seed)).genealogy
        @test curtail(whole, 30.0) == part
    end
    ## thinning the samples by p is equivalent to sampling at rate p*ψ
    sims(ψ, p) = map(1:400) do _
        h = thin(SimSEIR.simulate(; par..., ψ, time=25.0).genealogy, p)
        Float64[h.nsample, length(nodes(h)), length(roots(h))]
    end
    a, b = sims(0.1, 0.5), sims(0.05, 1)
    z = (sum(a)-sum(b)) ./ sqrt.(sum(x -> x.^2, a) .+ sum(x -> x.^2, b) .- (sum(a).^2 .+ sum(b).^2)/400)
    @test all(abs.(z) .< 4)

end

end
