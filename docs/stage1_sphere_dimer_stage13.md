# Stage 1.3 — Double-Sphere Lens Discovery Pipeline

Goal: discover small, structured image systems for the two-sphere “thick lens” using Kelvin ladders + rings, explore nearby geometries, and probe low-rank structure via axis sweeps + SVD.

## Geometry & specs
- Baseline inside spec: `specs/stage1_sphere_dimer_axis_point_inside.json` (R1=1, R2=1, d=2.4, z=1.2).
- Variants: left/right specs (charge outside) unchanged from Step 1.1.
- Geometry perturbations explored in Step 1.3: R2 ∈ [0.9,1.1], d ∈ [2.2,2.6] (non-intersecting).

## Basis (Kelvin ladder + rings)
- `sphere_kelvin_ladder`: deterministic Kelvin-inspired ladder, alternating inversions between spheres, capped at k_per_sphere ≤ 6 (inferred from n_candidates).
- `sphere_equatorial_ring`: 1–2 equatorial/nearsurface rings per sphere (z-axis normal, radius ~0.9R, optional z-shift toward gap).
- Basis types registered in `electrodrive/images/basis.py`; enabled via `basis_types` strings.

## Key tools
- `tools/stage1_sphere_dimer_random_explorer.py`: runs small grids/random configs on baseline geometry, evaluates vs oracle BEM (boundary + axis + gap belt). Outputs results.json, top_candidates.json under `runs/stage1_sphere_dimer/random_explorer/`.
- `tools/stage1_sphere_dimer_local_geometry_explorer.py`: perturbs R2,d around baseline, re-runs discovery with best config, evaluates, aggregates summary.json/top_candidates.json under `runs/stage1_sphere_dimer/local_geometry/`.
- `tools/stage1_sphere_dimer_axis_sweep_svd.py`: sweeps source z grid, runs discovery per z, builds weight matrix W, computes SVD, writes weights_svd.npz + summary.json under `runs/stage1_sphere_dimer/axis_sweep_svd/`.
- `tools/stage1_sphere_dimer_highres_diagnostics.py`: evaluates a discovered system vs oracle BEM on dense boundary/axis/gap grids; FMM flag stubbed for later.
- Discovery driver: `tools/stage1_sphere_dimer_discover.py` and config `configs/stage1_sphere_dimer_inside_kelvin.json`.

## BEM oracle
- Probe harness supports `--mode oracle` (h=0.2, max_refine=4, gmres_tol=1e-9, near-quadrature matvec on, fp64, GPU if available). Oracle summaries saved to `runs/stage1_sphere_dimer/oracle_all_summaries.json`.

## Tests
- Basis: `tests/test_images_basis_sphere_dimer.py`.
- Discovery smoke: `tests/test_stage1_sphere_dimer_kelvin_discovery_smoke.py`.
- BEM health: `tests/test_stage1_sphere_dimer_bem_health.py`.
Run with `PYTHONPATH=.` and reduced collocation env vars if needed.

## Outputs & layout
- Random explorer: `runs/stage1_sphere_dimer/random_explorer/` (per-run metrics + aggregated results/top_candidates).
- Local geometry: `runs/stage1_sphere_dimer/local_geometry/` (perturbed specs, metrics, top_candidates).
- Axis sweeps/SVD: `runs/stage1_sphere_dimer/axis_sweep_svd/<tag>/` (weights_svd.npz, summary.json).
- High-res diagnostics: `runs/stage1_sphere_dimer/highres/summary.json` plus diagnostics.npz.

## Next steps
- Identify “magic” candidates (low effective rank or very low errors with few images) from random/local exploration.
- Perform axis sweeps + SVD on top candidates, compare singular value decay.
- If exceptional, add a vault entry with geometry, system, metrics, and diagnostics.
