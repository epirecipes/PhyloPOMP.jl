## E03: validate the particle filters (Naive/Soft/Guided/Hard) against the exact
## likelihood from ExactSEIR on tiny problems, including cases designed to visit
## I == ℓ_I (hypothesis H1) and E == ℓ_E (hypothesis H2).
##
## For each case and variant: R independent pfilter runs with Np particles;
## logmeanexp of the likelihood estimates (unbiased on the natural scale) is
## compared with the exact log-likelihood; z = (lme - exact)/jackknife_se.
include("common.jl")

const R = parse(Int, get(ENV, "E03_R", "200"))
const NP = parse(Int, get(ENV, "E03_NP", "500"))
const SEED = 20261028
Random.seed!(SEED)

base = (β=3.0, σ=1.0, γ=0.5, ω=0.2, ψ=0.4, χ=0.0)

designed = [
    ## H1: a single infective carries the root lineage; transmissions from it occur
    ## with I == ℓ_I, so the naive "no colour change" transmission is never proposed.
    (name="H1_single_infective", newick="(([&&PhyloPOMP type=sample]:0.9)[&&PhyloPOMP type=sample]:0.6);",
     time=1.8, par=merge(base, (β=4.0, σ=0.7, pop=6, S0=4/6, E0=0.0, I0=1/6, R0=1/6))),
    ## H1 with a branch: both children must be infective-derived
    (name="H1_branch", newick="(([&&PhyloPOMP type=sample]:0.5,[&&PhyloPOMP type=sample]:0.8)[&&PhyloPOMP type=node]:0.5);",
     time=1.6, par=merge(base, (β=4.0, σ=0.8, pop=7, S0=5/7, E0=0.0, I0=1/7, R0=1/7))),
    ## H2: the root lineage can sit in the single exposed individual (E == ℓ_E)
    (name="H2_single_exposed", newick="([&&PhyloPOMP type=sample]:1.2);",
     time=1.5, par=merge(base, (β=3.0, σ=2.0, pop=6, S0=3/6, E0=1/6, I0=1/6, R0=1/6))),
]

## small simulated genealogies (kept only if lineage count stays small)
simulated = NamedTuple[]
let k = 0, tries = 0
    while length(simulated) < 4 && tries < 2000
        tries += 1
        par = merge(base, (pop=10, S0=0.7, E0=0.1, I0=0.1, R0=0.1))
        sim = SimSEIR.simulate(; par..., time=3.0)
        g = sim.genealogy
        ns = g.nsample
        (3 ≤ ns ≤ 6) || continue
        k += 1
        push!(simulated, (name="sim$(k)_n$(ns)", g=g, par=par))
    end
end

cases = vcat(
    [(name=c.name, g=parse_newick(c.newick, time=c.time), par=c.par) for c ∈ designed],
    simulated,
)

rows = NamedTuple[]
for c ∈ cases
    exact = ExactSEIR.loglik(c.g; c.par...)
    @printf("\n== %s (nodes=%d, samples=%d): exact logLik = %.4f\n",
            c.name, length(c.g), c.g.nsample, exact)
    for v ∈ (:naive, :soft, :guided, :hard)
        p = make_pomp(v, c.g; c.par...)
        t = @elapsed ll = pf_logliks(p, NP, R)
        s = summarize(ll)
        z = (s.lme - exact)/s.se
        @printf("  %-7s lme=% .4f ± %.4f  sd(ll)=%.3f  z=% .2f  finite=%d/%d  (%.1fs)\n",
                v, s.lme, s.se, s.sd, z, s.nfinite, s.n, t)
        push!(rows, (case=c.name, variant=String(v), exact=exact, lme=s.lme, se=s.se,
                     sd=s.sd, z=z, nfinite=s.nfinite, R=R, Np=NP, seconds=t))
    end
end
write_csv(joinpath(RESULTS, get(ENV, "E03_OUT", "e03_validate_filters.csv")), rows)
println("\nwrote ", get(ENV, "E03_OUT", "e03_validate_filters.csv"))
