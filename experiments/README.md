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
| `alive_filter.jl` | `AliveFilter` | weighted alive SMC with stopping-time normalization, explicit trial caps, and a multinomial bootstrap control |

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
| `e31_alive_filter.jl` | Does replenishing killed particles prevent collapse efficiently? | Higher-cap alive filtering completes 24/24 unguided large-N runs versus 3/24 systematic, but costs much more; no demonstrated fixed-precision gain. |

A lab notebook with the details, in OKF format, is kept outside the repository.

## E31: weighted alive particle filtering

On `feature/alive-filter`, stacked on `perf/colour-bookkeeping`:

```sh
julia +1.13 -t 4 --project=experiments experiments/test_alive_filter.jl
E31_PARTS=A julia +1.13 -t 4 --project=experiments experiments/e31_alive_filter.jl
E31_PARTS=BC julia +1.13 -t 4 --project=experiments experiments/e31_alive_filter.jl
E31_PARTS=D julia +1.13 -t 4 --project=experiments experiments/e31_alive_filter.jl
```

Part A compares systematic bootstrap, multinomial bootstrap, and alive filters,
both unguided and default-guided, against three tiny exact likelihoods: 64
replicates at 1000 particles. An additional unguided, two-particle root-colour
stress case uses 512 replicates per method. Part B uses the fitted 92- and
864-sample trees at 1000 particles; part C uses seed-27 population-1000/10000
simulations, guided and unguided, at 128 particles. B/C have 24 replicates per
cell, with methods interleaved in seeded random order after warm-up.
Part D repeats the unguided population-10000 condition with independent seeds
and a trial-cap multiplier of 1000, 50 times that of B/C. With no `E31_PARTS`
override, the script runs all four parts.

`E31_EXACT_REPS`, `E31_EXACT_NP`, `E31_RARE_REPS`, `E31_REPS`, `E31_NP`,
`E31_LARGE_NP`, `E31_POPS` and `E31_CAP_MULTIPLIER` override these defaults.
Guide construction and compilation warm-up are excluded from repeated timings.

Alive SMC draws independent ancestors from the previous weighted population.
Stop at `Np+1` positive-weight trials, discard the last survivor, and normalize
with `sum(retained weights)/(T-1)`, not with the number of successes or scheduled
trials. This extends the indicator-potential construction of
[Jasra et al.](https://arxiv.org/abs/1304.0151) to the existing nonconstant
FastSEIR importance weights. No population or colour propagation kernel changes.
The multinomial control isolates the cost of replenishment from the change in
ancestor sampling.

Refill batches shrink with the remaining survivor count, with a minimum target
of 64 trials to amortize threading. Any work beyond the stopping point is
ignored by the estimator but counted in `proposed`, `nevents` and runtime.
The per-interval trial cap is `20*(Np+1)` in B/C and at least `100*(Np+1)` in
the exact checks. A capped run raises an explicit `TrialLimit`: it has **no
likelihood estimate**, rather than a zero or partial likelihood. Raw CSVs mark
it `capped` with `NaN`; summary precision is suppressed for any cell with caps.
Ordinary zero-likelihood estimates remain `-Inf` and are included as zeros in
natural-scale means. Conditional survival probability must be positive for
uncapped termination; a finite ancestor cloud need not always satisfy this.
