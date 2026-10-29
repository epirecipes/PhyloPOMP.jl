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
| `resampling_filter.jl` | `ResamplingFilter` | exact every-event filtering with systematic, stratified, or residual-stratified ancestor selection |

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
| `e33_resampling.jl` | Do stratified or residual-stratified ancestors improve precision per second? | Exact target; no reliable gain. The initial fitted-tree residual efficiency estimate 4.02 becomes 0.50 in independent confirmation; both intervals include one. |

A lab notebook with the details, in OKF format, is kept outside the repository.

## E33: alternative ancestor resampling

On `feature/resampling-schemes`, stacked on `perf/colour-bookkeeping`:

```sh
julia +1.13 -t 4 --project=experiments experiments/test_resampling_filter.jl
E33_PARTS=A julia +1.13 -t 4 --project=experiments experiments/e33_resampling.jl
E33_PARTS=BC julia +1.13 -t 4 --project=experiments experiments/e33_resampling.jl
E33_PARTS=D julia +1.13 -t 4 --project=experiments experiments/e33_resampling.jl
```

Part A compares the canonical filter, a systematic research control, stratified
resampling and residual-stratified resampling against three tiny exact
likelihoods, guided and unguided: 64 replicates per cell at 1000 particles.
Part B uses the fitted 92-sample tree at 1000 particles and the fitted
864-sample tree at **4000** particles, to include a better-resolved precision
regime. Part C uses seed-27 population-1000/10000 simulations at 512 particles.
B/C each use 32 interleaved replicates per method and the default guide.
Part D is one independent-seed confirmation of residual-stratified versus
canonical filtering on the fitted 864-sample tree, using the same 4000-particle
budget and replicate count. It follows the initially promising but uncertain
natural-scale efficiency ratio. By default all four parts run.
Filter warm-up and guide construction are excluded from repeated timings.

`E33_EXACT_REPS`, `E33_EXACT_NP`, `E33_REPS`, `E33_NP`, `E33_TREE2_NP` and
`E33_LARGE_NP` override these defaults.

Stratified resampling draws one independent uniform point in each of Np equal
CDF strata. Residual-stratified resampling first assigns `floor(Np*W[i])`
offspring to each ancestor, then stratifies the remaining draws over the
normalized residual weights. All schemes have expected offspring counts
`Np*W[i]`, preserving the likelihood target. Epidemic and colour kernels are
reused unchanged. The systematic research control must reproduce canonical
likelihoods and event counts exactly at every recorded seed.

CSV summaries retain collapses as zero likelihoods (`-Inf` on the log scale).
The descriptive efficiency ratio is canonical `CV²*seconds` divided by the
candidate's, using natural-scale likelihood estimates; values above one favor
the candidate for averaging independent estimates. Its paired empirical
bootstrap interval uses 3000 resamples of replicate indices, including their
timings. These intervals do not resolve unobserved likelihood tails: interpret
them alongside relative Monte Carlo SE and the largest likelihood-sum share,
not as proof of time-to-fixed-precision improvement under particle scaling.
