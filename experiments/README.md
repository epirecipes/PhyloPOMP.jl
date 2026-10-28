# Scalability experiments

Scripts and results behind the scalability study of the exact SEIRS genealogy
likelihood. Everything targets Julia 1.13. The new package modules live in
`src/examples/`: `ExactSEIR` (exact likelihood for small problems), `SimSEIR`
(genealogy simulator) and `FastSEIR` (fast streaming filter).

## Setup

```sh
julia +1.13 --project=experiments -e 'import Pkg; Pkg.instantiate()'
julia +1.13 -t 8 --project=experiments experiments/e05_diagnostics.jl
```

E02 also calls an independent Python reference solver, `python/exact_mpmath.py`. It needs
`pip install mpmath`; set `PYTHON` to choose the interpreter. E11 uses
PiecewiseDeterministicMarkovProcesses.jl and OrdinaryDiffEq, both of which are
in the environment. Results are written to `results/` as CSV files.

## Scripts and main findings

| Script | Question | Finding |
|---|---|---|
| `e02_exact_crosscheck.jl` | Is `ExactSEIR` correct? | Agrees with the independent mpmath solver to ≤ 1e-15. |
| `e03_validate_filters.jl` | Are the filters unbiased? | Naive/Soft/Hard were biased low when I == ℓ_I ([#8](https://github.com/kingaa/PhyloPOMP.jl/issues/8)); unbiased after the fix. |
| `e03b_fast_validate.jl` | Is `FastSEIR` unbiased? | All support-correct options agree with the exact likelihood. |
| `e04_first_principles.jl` | Does the likelihood match forward simulation? | P(0 samples) and P(1 sample) agree (\|z\| < 0.7). |
| `e05_diagnostics.jl` | Where do zero weights come from, and what does precision cost? | Applying events before resampling plus guided proposals: zero weights 43% → 8%; about 12× less time to SD 1 than GuidedSEIR. |
| `e06_apf.jl`, `e07_variance.jl` | Does look-ahead (APF) resampling help? | Exact, but gives no variance reduction on `seir_trees[1]`. |
| `e08_threads.jl` | Threading | 2.8–3.5× on 8 threads; bitwise reproducible across thread counts. |
| `e09_tree2_probe.jl` | The 864-sample tree | All filters degenerate at Np ≤ 4000. |
| `e10_scaling.jl` | Scaling with tree and population size | Cost is linear in population events; `FastSEIR` is 4–60× faster to SD 1. |
| `e11_hard_pdmp.jl`, `pdmp_proposals.jl` | Time-varying proposals (PDMP.jl) and an ODE population twist | Exact; removes zero weights but not variance; about 100× more costly. |
