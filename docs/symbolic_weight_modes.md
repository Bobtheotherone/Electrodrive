# Symbolic weight-mode extraction (Stage-0/1 axis sweeps)

This repo now supports the 4.10 pipeline for extracting human-readable laws from axis sweeps:

- Sweep source z-positions for a fixed geometry and basis.
- Stack discovered weight vectors into a padded matrix `W[K, M]`.
- Compute the SVD (`U, S, VT`) and keep the top `r` modes (`S[i] * VT[i, :]`).
- Fit lightweight symbolic laws to each mode curve (`w_i(z)`), preferring low-degree polynomials or tiny rationals.
- Persist audit artifacts:
  - `weights_vs_axis.npy` (padded weight matrix),
  - `svd_modes.npy` (dict: `U, S, VT, z_grid, mode_curves, sigma_norm, effective_rank`),
  - `symbolic_fits.json` (mode fits + errors),
  - `metrics.json` (rank/reconstruction stats),
  - `summary.md` (geometry + research_wishlist).

## CLI entry points

### Stage-0 grounded sphere (axis charge)

```
python tools/stage0_sphere_axis_moi.py ^
  --z 1.25 1.5 2.0 ^
  --nmax 1 2 3 ^
  --basis sphere_kelvin_ladder ^
  --adaptive-collocation-rounds 2 ^
  --max-rank 3 --max-poly-degree 4 ^
  --lambda-weight-mode 0.0 ^
  --vault
```

Artifacts land under `runs/stage0/sphere_moi/symbolic_n<N>/`. Vault copies live in `the_vault/<slug>/` when `--vault` is set.

### Stage-1 sphere dimer axis sweep

```
python tools/stage1_sphere_dimer_axis_sweep_svd.py ^
  --spec specs/stage1_sphere_dimer_axis_point_inside.json ^
  --basis sphere_kelvin_ladder,sphere_equatorial_ring ^
  --z 0.3 0.7 1.0 1.3 1.6 1.9 2.1 ^
  --nmax 8 --reg-l1 1e-3 ^
  --max-rank 3 --max-poly-degree 4 ^
  --lambda-weight-mode 0.0 ^
  --vault
```

Outputs land in `runs/stage1_sphere_dimer/axis_sweep_svd/`.

## Weight-mode controller (optional feedback)

- Provide `--mode-dir <path>` containing `svd_modes.npy` and `symbolic_fits.json`.
- Enable with `--use-weight-modes`; prior strength is `--lambda-weight-mode` (quadratic pull toward predicted weights).
- Safety gates:
  - spectral gap: `sigma[r]/sigma[0] < --spectral-gap-thresh` (default 0.1),
  - fit quality: `rel_rmse` for each mode < `--rel-rmse-thresh` (default 0.2).
- When enabled, a predicted weight vector `w_pred(z)` is reconstructed from fitted modes and used as a quadratic prior inside ISTA/LISTA/AL paths.

## Research wishlist hooks

Each summary includes a `research_wishlist` section; common follow-ups:
- Derive spectral gaps along the axis from the boundary integral operator.
- Map rational-fit poles/zeros to candidate ladder distances.
- Extend the controller beyond the axis (off-axis collocation / harmonic lifts).
