# Canonical Stage-0/Stage-1 Specs

- **Stage-0 (single grounded sphere, external axis charge)**  
  `specs/sphere_axis_point_external.json` — canonical for MOI training, Stage-0 harness, and regression.
- **Stage-1 (grounded sphere dimer, midpoint charge inside lens)**  
  `specs/stage1_sphere_dimer_axis_point_inside.json` — canonical for Stage-1 dimer discovery/regression.

Both are exposed via `electrodrive.orchestration.spec_registry`:

- `stage0_sphere_external_path()`, `load_stage0_sphere_external()`
- `stage1_sphere_dimer_inside_path()`, `load_stage1_sphere_dimer_inside()`
- Variant enumerators: `list_stage0_variants()`, `list_stage1_variants()`

## Where the canonicals are used
- Stage-0 bilevel training (`electrodrive.images.training.train_stage0`) now loads `load_stage0_sphere_external`.
- Stage-0 MOI/BEM harnesses (`tools/stage0_sphere_bem_vs_analytic.py`, `tools/stage0_sphere_axis_moi.py`) share the same path.
- Stage-1 dimer tools (`tools/stage1_sphere_dimer_*`) default to `stage1_sphere_dimer_inside_path`.
- Basis/regression tests exercising Kelvin ladders and dimer rings now pull specs from `spec_registry`.

## New variants (for meta-learning sweeps)
- `specs/stage0_sphere_axis_point_external_r1.5.json`: R=1.5 sphere, charge at 1.5 radii (z=2.25); wider scale separation for Stage-0 sampling.
- `specs/stage1_sphere_dimer_axis_point_inside_D2.5.json`: d=2.5 separation (vs. 2.4 baseline), charge at midpoint z=1.25.

Guidance:
- Keep canonicals unchanged; add new variants alongside them.
- Prefer descriptive filenames (`r1.5`, `D2.5`, etc.) and register them via `spec_registry` when they should participate in meta-learning datasets.
