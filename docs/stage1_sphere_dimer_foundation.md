# Stage 1 — Sphere Dimer Foundation (Step 1.1)

Baseline geometry: two grounded spheres on the z-axis with an axis-aligned unit point charge.

- Sphere 1: center (0, 0, 0), radius R1 = 1.0, grounded.
- Sphere 2: center (0, 0, d), radius R2 = 1.0, grounded, with d = 2.4 (no overlap).
- Charge: q = 1.0 at (0, 0, z) with regimes:
  - **Inside**: z = 1.2 (midpoint between spheres).
  - **Left/outside**: z = -0.5.
  - **Right/outside**: z = 2.9.

## Canonical specs

- `specs/stage1_sphere_dimer_axis_point_inside.json`
- `specs/stage1_sphere_dimer_axis_point_left.json`
- `specs/stage1_sphere_dimer_axis_point_right.json`

Fields: `domain="R3"`, `BCs="Dirichlet"`, two `sphere` conductors (id 0,1), one point charge, `symmetry=["axis"]`, and a bounding box in `domain_meta`.

## BEM configuration (probe defaults)

- `use_gpu` if available, `fp64=True`
- `initial_h=0.25`, `max_refine_passes=3`
- `use_near_quadrature=True`, `use_near_quadrature_matvec=True`, `near_quadrature_order=2`, `near_quadrature_distance_factor=2.0`
- `gmres_tol=1e-8`, `vram_autotune=False`
- Seeds fixed to 1234 for reproducibility.

## Probe harness

- Script: `tools/stage1_sphere_dimer_bem_probe.py`
- Usage example:
  ```bash
  PYTHONPATH=. python tools/stage1_sphere_dimer_bem_probe.py \
      --spec specs/stage1_sphere_dimer_axis_point_inside.json \
      --out runs/stage1_sphere_dimer/bem_probe_inside
  ```
- Ladder of attempts: GPU/CPU, fp64/fp32, coarse refinement fallback.
- Health classification reused from `_bem_probe.py` (`ok|warn|fail`).
- Outputs per spec: `runs/stage1_sphere_dimer/bem_probe_<spec>/summary.json` plus `all_summaries.json`.

## Tests

- `tests/test_stage1_sphere_dimer_bem_health.py`:
  - Runs the probe for all three specs.
  - Asserts mesh non-degeneracy (`n_panels>0`, `total_area>0`), finite `bc_residual_linf`, GMRES success, and overall `ok|warn`.
  - Use `PYTHONPATH=.` when invoking pytest.

## Notes / Next steps

- This step establishes BEM health and geometry definitions only. No image-basis discovery, axis sweeps, or SVD analyses are included yet.
- Future Stage 1 steps can hook into the probe outputs for higher-resolution diagnostics or FMM backends.

## Step 1.2 — Kelvin Ladder & Ring Basis (overview)

- Basis types:
  - `sphere_kelvin_ladder`: finite Kelvin-inspired ladder of point images per sphere (k per sphere inferred from `n_candidates`, capped at 6).
  - `sphere_equatorial_ring`: 1–2 equatorial/nearsurface rings per sphere (z-axis normal), radii slightly inside the surface.
- Discovery config:
  - Example config: `configs/stage1_sphere_dimer_inside_kelvin.json`.
  - Discovery driver: `tools/stage1_sphere_dimer_discover.py` (defaults to inside spec, basis `sphere_kelvin_ladder,sphere_equatorial_ring`, nmax=8, reg_l1=1e-3).
- Probe refinement:
  - Oracle-grade probe available via `python tools/stage1_sphere_dimer_bem_probe.py --mode oracle --out runs/stage1_sphere_dimer`.
- Smoke discovery:
  - Run with reduced collocation for speed, e.g. `EDE_IMAGES_N_POINTS=256 EDE_IMAGES_RATIO_BOUNDARY=0.8 PYTHONPATH=. python tools/stage1_sphere_dimer_discover.py --out runs/stage1_sphere_dimer/discover_smoke`.
