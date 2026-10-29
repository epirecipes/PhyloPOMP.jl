# Inference experiments

Experiments on improving inference with the SEIRS genealogy likelihood, run on the
`inference-experiments` branch. That branch is `master` plus the reviewed fork
branches: `FastSEIR` (fast filter), `ExactSEIR` (exact likelihood for small problems),
`SimSEIR` (genealogy simulator), the transmission-support fix, and
`feature/fast-mif`, which packages E19's iterated filtering as `mif` on a
`FastSEIR.Tree`. Everything targets Julia 1.13.

The research modules here reuse `FastSEIR`'s internals and are not part of the
package.

| File | Module | Purpose |
|---|---|---|
| `common.jl` | — | default parameters, ad hoc and equilibrium-matched guides, helpers, CSV output |
| `apf_filter.jl` | `APF` | next-event look-ahead auxiliary particle filter |
| `weight_components.jl` | `WeightDiag` | per-component decomposition of the particle weights |
| `weighted_filter.jl` | `WFilter` | weighted-rate and twisted (Doob-style) proposals |
| `fast_mif.jl` | `FastMif` | iterated filtering (IF2) with per-particle parameters; the prototype of the package's `mif(tr::Tree; ...)` |
| `nested_colour_filter.jl` | `NestedColour` | conditional colour SMC sharing a population trajectory; exact likelihood target, non-destructive sampling only |

## Setup

```sh
julia +1.13 --project=experiments -e 'import Pkg; Pkg.instantiate()'
julia +1.13 -t 8 --project=experiments experiments/e16_weight_components.jl
```

Results are written to `results/` as CSV files. Replicate and particle numbers can be
changed through the environment variables read at the top of each script.

## Scripts and main findings

| Script | Question | Finding |
|---|---|---|
| `e12_guide.jl` | How should the guide be tuned? | Conductance c ≥ 2 is needed. The best tree-1 cell (π_E = 0.5, c = 4; SD 0.50 vs 0.61) is worse on tree 2. |
| `e13_crn_profile.jl` | Do common random numbers smooth likelihood slices? | No: neighbouring estimates are only ≈0.2 correlated, because resampling is discontinuous in the parameters. |
| `e14_tree2_diagnosis.jl` | Where is the variance on `seir_trees[2]`? | Everywhere: ≈0.03 per interval over ~1560 intervals; ESS/Np ≈ 0.16; 15% of particles die per event. |
| `e15_apf.jl` | Does a next-event look-ahead help? | Exact. It raises ESS but not precision on tree 1; on tree 2 the gain is small. |
| `e16_weight_components.jl` | Which part of the weight varies? | On tree 2, the colour-compatibility jump factors Σφ (variance 11.8 of 20.5), then the guide ratio and parent-in-E deaths. |
| `e17_weighted.jl`, `e18_twisted.jl` | Move Σφ, or the guide ratio, from jump weights into the rates? | Both are exact. Moving Σφ lowers the tree-2 mean log-likelihood by 60–90. A guide-only Doob twist with the matched guide removes zero weights and helps only at small Np. |
| `e19_mif.jl` | Can FastSEIR drive IF2 and profile likelihoods? | Yes: seconds per fit on a 92-sample tree. The MCAP β CI [2.23, 5.86] covers the truth (4). |
| `e20_scaling.jl` | How many particles does `seir_trees[2]` need at the defaults? | It is pre-asymptotic below ~64k. At 64k, SD is 3.0 in 81 s (8 threads); SD ≈ 1 needs ≈560k particles. |
| `e21_tree2_mif.jl`, `e21b_refine.jl` | Do the default parameters fit `seir_trees[2]`? | No. IF2 raises the log-likelihood by ≈2,080 units (β≈2.8, σ≈0.6, γ≈0.48, ω≈5.2, ψ≈0.027, pop≈173). There the plain filter has SD 0.68 at Np = 4000 in 6 s, so the tree-2 degeneracy above was model misfit. |
| `e23_colour.jl` | Can the colour proposal cost O(1) per population event? | Yes. Freezing non-parent guide ratios mid-interval and keeping running sums is exact and 1.9–2.9× faster on `seir_trees[2]` and a lineage-dense tree, at equal precision. |
| `e29_nested_colour.jl` | Does averaging colours over shared populations improve inference efficiency? | K = 4/16 makes runs roughly 1.3–2.3 times cheaper, but no fixed-precision gain is established: moderate-N variance offsets the saving, and large-N likelihood averages remain dominated by rare estimates. |

A lab notebook with the details, in OKF format, is kept outside the repository.

## E29: nested conditional colour filters

Branch `feature/nested-colour-filter` builds on `perf/colour-bookkeeping`.
`NestedColour.pfilter_nested(tr; Np=128, K=4, params...)` carries `Np` population
particles, each with `K` conditional colour particles. It simulates the full
population rates once per group. Incompatible recoveries kill colours rather
than changing the shared population rate; no extra recovery killing potential
is added. Both levels' resampling normalizers contribute to the likelihood.

This is finite inner SMC, not exact analytic Rao--Blackwellization or a
factorized colour approximation. Only `chi=0` is supported. It is research code,
does not estimate sufficient statistics, and changes no package API.

```sh
julia +1.13 -t 4 --project=experiments experiments/test_nested_colour.jl

# A: three exact-reference cases, 64 replicates per method and guide.
E29_PARTS=A julia +1.13 -t 4 --project=experiments experiments/e29_nested_colour.jl

# B: N=1000 and N=10000, 24 replicates, with/without the default guide.
E29_PARTS=B julia +1.13 -t 4 --project=experiments experiments/e29_nested_colour.jl

# C: an equilibrium-matched guide on the same N=10000 genealogy.
E29_PARTS=C julia +1.13 -t 4 --project=experiments experiments/e29_nested_colour.jl
```

Part A compares FastSEIR and nested K=1/4/16, plus K=4 with inner ESS checked
every population event, against `ExactSEIR`. All 30 comparisons agree within
2.57 Monte Carlo standard errors. Part B holds the total colour budget at
512: ordinary FastSEIR uses 512 independent populations, nested K=4 uses 128,
and K=16 uses 32. The two simulated genealogies reproduce E27's seed-27,
time-40 design with sampling rate `8/N`; evaluation is at the generating
parameters, not fitted estimates. Every method uses E23's optimized proposal.
Methods are interleaved in seeded random order. Part C uses independent
replicate seeds and the existing equilibrium-matched guide, not a new guide.

`E29_REPS`, `E29_BUDGET`, `E29_POPS`, and `E29_MODES` override the main settings;
`E29_EXACT_REPS` and `E29_EXACT_BUDGET` control part A. Colour budgets must be
divisible by 16. The kernel defaults to checking inner ESS every 32 population
events and resampling below half the inner particle count, plus mandatory
resampling at observed-event boundaries.

Raw CSVs retain collapsed runs as `-Inf`. Summaries report finite-run counts,
natural-scale log means, relative Monte Carlo SEs, and the largest run's share
of the likelihood sum. They do not silently discard collapses. At N=10000,
two of 24 unguided runs collapse for every method, and even guided averages
are dominated by individual runs. The matched-guide follow-up still has
log-likelihood SDs of 8--10 and maximum shares of 52--100%. These data do not
establish a reliable likelihood-precision improvement despite lower runtime.
