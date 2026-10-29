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

    @info h2("iterated filtering")
    ## start where the likelihood is ≈ 9 log units below its maximum
    start = (β=8.0, γ=0.5)
    kernel = @perturbn β ~ LogNormal(0.02) γ ~ LogNormal(0.02)
    fit(seed) = begin
        seed!(seed)
        mif(tr; Np=500, Nmif=10, perturbations=kernel, cooling=geometric_cooling(0.5),
            start..., rng=Xoshiro(seed))
    end
    mf = fit(5)
    @test mf isa FastSEIR.FastMifd
    @test length(mf.trace) == 11 && length(mf.logLik) == 10
    @test all(isfinite, mf.logLik)
    @test size(traces(mf)) == (11, 13)
    @test coef(mf).σ == 1.0 && coef(mf).pop ≈ 100
    @test coef(mf).β < start.β && coef(mf).γ > start.γ
    ll(θ) = logmeanexp([logLik(pfilter(tr; Np=500, θ..., rng=Xoshiro(i))) for i ∈ 1:5])
    @test ll((β=coef(mf).β, γ=coef(mf).γ)) > ll(start) + 4
    ## the perturbations use the global generator, so seed both
    @test coef(fit(5)) == coef(mf)

    @info h2("expectation-maximization")
    s = pfilter(tr, Np=500, rng=Xoshiro(23)).suffstats
    ## every path contains all the samples (none destructive, since χ = 0) and
    ## a transmission at each branch point
    @test s.samples ≈ count(==(PhyloPOMP.Sample), tr.type)
    @test s.destructive == 0
    @test s.transmissions > count(==(PhyloPOMP.Node), tr.type)
    ef = FastSEIR.em(tr; Np=1000, Nem=10, estimate=(:β,:γ), start..., rng=Xoshiro(6))
    @test ef isa FastSEIR.FastEMd
    @test length(ef.trace) == 11 && length(ef.logLik) == 10
    @test size(traces(ef)) == (11, 13)
    @test coef(ef).σ == 1.0 && coef(ef).pop == 100
    ## from the poor start, EM raises the recovery rate towards its estimate
    @test coef(ef).γ > start.γ + 0.03

    @info h2("fixed-lag sufficient statistics")
    ## Hand-computed ancestry with a duplicated parent, an identity step,
    ## a permutation, ring-buffer wraparound, and an unfinished final tail.
    history = FastSEIR.LaggedStats(2,1)
    stats = zeros(length(FastSEIR.STATS),2)
    for (step, (values, weights, parents)) ∈ enumerate((
        ((1.0,10.0), [0.25,0.75], [2,2]),
        ((12.0,14.0), [0.6,0.4], [1,2]),
        ((15.0,19.0), [0.2,0.8], [2,1]),
        ((26.0,26.0), [0.75,0.25], [1,2]),
    ))
        stats[1,:] .= values
        FastSEIR.lagged_stats!(history,stats,weights,step,step==4)
        history.parents[:,mod1(step,2)] .= parents
        history.previous .= stats[:,parents]
    end
    @test history.total[1] ≈ 10 + 3.6 + 4.5 + 8
    @test all(iszero,history.total[2:end])
    online = FastSEIR.LaggedStats(2,0)
    stats[1,:] .= (1.0,10.0)
    FastSEIR.lagged_stats!(online,stats,[0.25,0.75],1,false)
    online.previous .= stats
    stats[1,:] .= (3.0,14.0)
    FastSEIR.lagged_stats!(online,stats,[0.6,0.4],2,true)
    @test online.total[1] ≈ 7.75 + 2.8

    baseline = pfilter(tr; Np=200, rng=Xoshiro(28))
    for lag ∈ (0,1,7,length(tr.time),typemax(Int))
        lagpf = pfilter(tr; Np=200, smoothing_lag=lag, rng=Xoshiro(28))
        @test logLik(lagpf) == logLik(baseline)
        @test lagpf.nevents == baseline.nevents
        @test cond_logLik(lagpf) == cond_logLik(baseline)
        @test lagpf.suffstats.samples ≈ count(==(PhyloPOMP.Sample),tr.type)
        @test lagpf.suffstats.destructive == 0
        @test all(isfinite,FastSEIR.score(lagpf; β=4, σ=1, γ=1, ω=1, ψ=0.02, χ=0))
        lag ≥ length(tr.time) && @test lagpf.suffstats == baseline.suffstats
    end
    @test_throws AssertionError pfilter(tr; smoothing_lag=-1)
    for trigger ∈ (0.0,0.5,1.0)
        lagpf = pfilter(tr; Np=200, smoothing_lag=7, trigger, rng=Xoshiro(29))
        serial = pfilter(tr; Np=200, smoothing_lag=7, trigger, rng=Xoshiro(29), threaded=false)
        @test lagpf.suffstats == serial.suffstats
        @test logLik(lagpf) == logLik(pfilter(tr; Np=200, trigger, rng=Xoshiro(29)))
    end
    initial = parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8)
    initial[2].slate = 0.0
    initialpf = pfilter(FastSEIR.Tree(initial,m); Np=300, par..., smoothing_lag=0, rng=Xoshiro(30))
    @test isfinite(logLik(initialpf))
    @test initialpf.suffstats.samples ≈ 2

end

end
