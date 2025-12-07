Inner-rim eigen-image experiment for thin torus (numerical conjecture)

Geometry
- Grounded torus R=1.0, a=0.15; charge q=1 at (0,0,0.6).

Runs and hyperparameters
- Baseline: basis [point, toroidal_eigen_mode_boundary], n_max=12, reg_l1=1e-3, boundary_weight=0.8, per_type_reg {point:2e-3, toroidal_eigen_mode_boundary:8e-4}.
- Hybrid: baseline + inner_rim_ribbon, n_max=12, reg_l1=8e-4, same boundary_weight, per_type_reg {point:2e-3, toroidal_eigen_mode_boundary:8e-4, inner_rim_ribbon:8e-4}. Elements ~4 eigen modes + 1 ribbon + 7 points (12 total).
- Refined (rich_inner_rim): added narrower arcs triggered by near-contact rule; basis [point, toroidal_eigen_mode_boundary, inner_rim_arc, rich_inner_rim], n_max=12, reg_l1=8e-4 (unstable; see below).

Key metrics
- Stage4 (collocation-based, belts on inner rim):
  * Baseline: boundary_mae 1.516e8; offaxis_rel 7.20; belt_rel 2.65; n_images 12.
  * Hybrid ribbon: boundary_mae 1.533e8; offaxis_rel 0.905; belt_rel 2.24; n_images 12.
  * Refined rich_inner_rim: boundary_mae 1.578e8; offaxis_rel 1.12; belt_rel 0.713; n_images 12 (4 eigen, 4 arcs, 4 points).
- BEM diagnostics (tools/diagnose_torus_errors.py, 200x200 r–z grid):
  * Baseline: inner_mean_abs 1.29e8, inner_mean_rel 2.25, max_abs 1.24e10, mean_rel 2.54.
  * Hybrid ribbon: inner_mean_abs 7.22e7, inner_mean_rel 1.19, max_abs 1.15e10, mean_rel 2.73.
  * Refined rich_inner_rim: inner_mean_abs 6.51e10, inner_mean_rel 9.09e2, max_abs 6.25e12 (unstable despite good belt_rel in Stage4).

Interpretation
- Adding a single inner_rim_ribbon alongside 3–4 global eigen modes trims the inner-rim belt error (Stage4 belt_rel 2.65?2.24) and lowers inner-rim BEM errors (~44% drop in inner_mean_abs, ~47% drop in inner_mean_rel). Boundary_mae remains similar; overall mean_rel is slightly worse, so the benefit is localized to the inner rim.
- The refined rich_inner_rim arcs (narrow span near sigma˜p) improved collocation belt_rel but produced huge BEM errors, indicating numerical instability/overfitting; treat as negative result.

Files
- spec.json: canonical thin torus spec.
- discovered_system_hybrid.json: hybrid ribbon system; discovered_system_baseline.json: baseline eigen-only.
- metrics.json: consolidated Stage4 + diagnostics for baseline, hybrid, and refined run.
- Diagnostics NPZ/PNGs: thin_baseline_eigen.*, thin_hybrid_ribbon.*, thin_refined_rich_inner.*.

Status
- Hybrid ribbon system is a numerical conjecture: small localized primitive plus global eigen modes reduces inner-rim boundary-layer error while keeping n_images=12. Further tuning needed to improve global mean error and stability; refined arcs currently unstable.
