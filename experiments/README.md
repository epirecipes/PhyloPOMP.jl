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
| `particle_allocation.jl` | `ParticleAllocation` | exact filtering with predetermined interval particle counts and independent-pilot budget planning |

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
| `e32_particle_allocation.jl` | Does an independent pilot allocate particle work better across intervals? | Exact target; no established efficiency gain. Large-N log SD falls from 8.22 to 5.85, but one allocated run supplies 95% of the likelihood sum. |

A lab notebook with the details, in OKF format, is kept outside the repository.

## E32: independent-pilot particle allocation

On `feature/particle-allocation`, stacked on `perf/colour-bookkeeping`:

```sh
julia +1.13 -t 4 --project=experiments experiments/test_particle_allocation.jl
E32_PARTS=A julia +1.13 -t 4 --project=experiments experiments/e32_particle_allocation.jl
E32_PARTS=BC julia +1.13 -t 4 --project=experiments experiments/e32_particle_allocation.jl
```

Part A compares the canonical filter, a uniform-count research control and a
250/2000-particle alternating schedule against three tiny exact likelihoods,
guided and unguided, with 64 replicates per cell and uniform baseline 1000.
Part B uses the fitted 92- and 864-sample trees with uniform baseline 1000.
Part C uses seed-27 population-1000/10000 simulations with uniform baseline 512.
B/C each have 32 interleaved held-out replicates per method, using the default
guide. Eight independent 256-particle pilot runs learn one frozen schedule
per condition. Compilation warm-up is excluded.

`E32_EXACT_REPS`, `E32_EXACT_NP`, `E32_REPS`, `E32_NP`, `E32_LARGE_NP`,
`E32_PILOT_REPS` and `E32_PILOT_NP` override these defaults.

For interval k, estimate the variance proxy as the pilot average of
`Npilot*(Npilot/ESS[k]-1)`. The work proxy per particle is population events
per pilot particle, plus the number of colour words and one, plus the initial
root count on the first active interval. Choose counts proportional to
`sqrt(variance_proxy/work_proxy)`, with bounds near 1/8 and 8 times the uniform
count and rounding to multiples of 16. The continuous proxy budget matches
uniform; rounding can change it slightly. Neither the work proxy nor local
ESS is claimed to estimate full likelihood variance or actual wall time.

Schedules are chosen independently of held-out draws. Unequal-size systematic
resampling preserves expected offspring proportions, and likelihood increments
use the current particle count. Population and colour kernels are unchanged.
The uniform research control isolates wrapper overhead from allocation effects.

Raw CSVs retain `-Inf` collapses. Natural-scale means and largest-contribution
diagnostics accompany log SD. Pilot costs are separate, with amortization over
the held-out replicate count also reported. `*_schedule.csv` saves every count
and planning proxy; `*_pilot.csv` saves interval diagnostics. Its total
`replicate_sec` and log likelihood repeat across intervals: count each pilot
replicate once when aggregating those two fields.
