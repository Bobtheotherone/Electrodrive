# Experiment Manifest

This sandbox captures reproducible diagnostics for the FMM stack. All artifacts live under `_agent_sandbox/experiments`.

## Families

- **local_rescaling_sweep**: Audits `_rescale_locals_packed` against the canonical `ratio^{-(l+1)}` rule. Sweeps multiple ratios and seeds, logging per-order relative error and bias.
- **l2l_translation**: Compares operator `l2l` to explicit P2L at the child center. Sources live on an external shell to guarantee convergence. Includes local potential vs direct potential at random child targets.
- **future_fmm_large_n** (planned): Grid over `n_points` × geometry modes to reproduce stress-test failures (`rel_l2_err` drift). Will reuse LaplaceFmm3D or tree-level operators once a stable harness is in place.
- **l2l_sweep**: Maps L2L accuracy and high-`l` bias versus expansion order `p` and normalized shift `|t|/scale`. For each combination, draws multiple random directions on S², computes per-order relative error and bias, and compares local potentials to direct fields near the child box.
- **m2l_sweep**: Mirrors `l2l_sweep` but probes M2L. Builds multipoles from shell sources, translates to locals at various separations, and compares coefficients and potentials against explicit P2L/direct references over `p`, `|t|/scale`, and random directions.
- **fmm_stress_grid**: End-to-end LaplaceFmm3D runs mirroring `test_stress`. Sweeps over N, geometry mode, expansion order, theta, and leaf size; records global rel_l2_error and per-target error quantiles.
- **l2l_usage_stats**: Extracts the actual `(p, |t|/scale, level)` distribution of L2L translations from a built LaplaceFmm3D tree (e.g., stress-test setups) to overlay against sweep error maps.

## Execution

Run experiments from repo root:

```bash
python electrodrive/fmm3d/_agent_sandbox/experiments/run_experiments.py local_rescaling --p 8 --ratios 0.25 0.5 2.0 4.0 --n-realizations 3
python electrodrive/fmm3d/_agent_sandbox/experiments/run_experiments.py l2l --p 8 --n-src 64 --translation 0.3 0.2 -0.1
# L2L sweep example
python electrodrive/fmm3d/_agent_sandbox/experiments/run_experiments.py l2l_sweep --p-list 4 8 --n-src 64 --scale-list 1.0 --translation-norm-list 0.25 0.5 0.75 --n-directions 3 --seed 40000
# M2L sweep example
python electrodrive/fmm3d/_agent_sandbox/experiments/run_experiments.py m2l_sweep --p-list 4 6 8 --translation-norm-list 1.0 1.5 2.0 --n-directions 2 --seed 60000
# FMM stress grid (mirrors test_stress)
python electrodrive/fmm3d/_agent_sandbox/experiments/run_experiments.py fmm_stress_grid --n-points-list 512 1024 2048 --mode-list uniform clusters --expansion-orders 4 6 8 --theta-list 0.5 --max-leaf-size-list 64 --seed 123
# L2L usage extraction from built trees
python electrodrive/fmm3d/_agent_sandbox/experiments/fmm_tree_usage.py --n-points 2048 --mode uniform --expansion-order 8 --theta 0.5 --max-leaf-size 64 --seed 123 --tag stress_2048_uniform
```

Each run appends one JSON object to `_agent_sandbox/experiments/results/*.jsonl` with seeds, parameters, spectral metrics, and git commit.

Outputs specific to L2L sweep:
- JSONL: `_agent_sandbox/experiments/results/l2l_sweep.jsonl`
- Summary CSV: `_agent_sandbox/experiments/results/l2l_sweep_summary.csv`

Outputs specific to M2L sweep:
- JSONL: `_agent_sandbox/experiments/results/m2l_sweep.jsonl`

Outputs specific to FMM stress grid:
- JSONL: `_agent_sandbox/experiments/results/fmm_stress_grid.jsonl`

Outputs specific to L2L usage mapping:
- JSONL: `_agent_sandbox/experiments/results/l2l_usage_stats.jsonl`
