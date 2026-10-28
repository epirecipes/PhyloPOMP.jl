## E03b: validate FastSEIR proposals/orderings against the exact likelihood, and
## isolate H2 (GuidedSEIR's `+1` applied to progressions) on a stress case.
include("common.jl")
using PhyloPOMP: FastSEIR

const R = parse(Int, get(ENV, "E03_R", "200"))
const NP = parse(Int, get(ENV, "E03_NP", "500"))
Random.seed!(20261029)

base = (β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.4, χ=0.0)
mkguide(g) = guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!)

cases = [
    (name="H1_single_infective", g=parse_newick("(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);", time=1.8),
     par=merge(base, (β=4.0, σ=0.7, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6))),
    (name="H1_branch", g=parse_newick("(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);", time=1.6),
     par=merge(base, (β=4.0, σ=0.8, pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7))),
    (name="destructive", g=parse_newick("(([&&PhyloPOMP type=sample]:0.6,[&&PhyloPOMP type=sample]:1.1)[&&PhyloPOMP type=node]:0.4);", time=2.0),
     par=merge(base, (χ=0.3, pop=6, S0=0.5, E0=1/6, I0=2/6, R0=0.0))),
]
## add the simulated E03 trees (same seed/protocol)
let k = 0, tries = 0
    Random.seed!(20261028)
    while k < 4 && tries < 2000
        tries += 1
        par = merge(base, (pop=10, S0=0.7, E0=0.1, I0=0.1, R0=0.1))
        g = SimSEIR.simulate(; par..., time=3.0).genealogy
        (3 ≤ g.nsample ≤ 6) || continue
        k += 1
        push!(cases, (name="sim$(k)_n$(g.nsample)", g=g, par=par))
    end
end

configs = [
    (label="naive/pre/unguided", opt=FastSEIR.Options(proposal=:naive, guided=false, order=:pre)),
    (label="fixed/pre/unguided", opt=FastSEIR.Options(proposal=:fixed, guided=false, order=:pre)),
    (label="fixed/post/guided",  opt=FastSEIR.Options(proposal=:fixed, guided=true,  order=:post)),
    (label="local/pre/unguided", opt=FastSEIR.Options(proposal=:local, guided=false, order=:pre)),
    (label="local/post/unguided",opt=FastSEIR.Options(proposal=:local, guided=false, order=:post)),
    (label="local/post/guided",  opt=FastSEIR.Options(proposal=:local, guided=true,  order=:post)),
    (label="local/post/guided/ess", opt=FastSEIR.Options(proposal=:local, guided=true, order=:post, resample=:ess, trigger=0.5)),
]

rows = NamedTuple[]
for c ∈ cases
    exact = ExactSEIR.loglik(c.g; c.par...)
    tr = FastSEIR.Tree(c.g, mkguide(c.g))
    @printf("\n== %s: exact = %.4f\n", c.name, exact)
    for cf ∈ configs
        t = @elapsed ll = FastSEIR.logliks(tr, c.par; Np=NP, n=R, opt=cf.opt)
        s = summarize(ll)
        z = (s.lme - exact)/s.se
        @printf("  %-22s lme=% .4f ± %.4f sd=%.3f z=% .2f (%.2fs)\n", cf.label, s.lme, s.se, s.sd, z, t)
        push!(rows, (case=c.name, config=cf.label, exact=exact, lme=s.lme, se=s.se, sd=s.sd, z=z, seconds=t))
    end
end
write_csv(joinpath(RESULTS, "e03b_fast_validate.csv"), rows)
