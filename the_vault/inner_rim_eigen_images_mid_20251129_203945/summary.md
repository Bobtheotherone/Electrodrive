Inner-rim eigen-image experiment for mid torus (numerical conjecture / negative result)

Geometry
- Grounded torus R=1.0, a=0.35; charge q=1 at (0,0,0.7).

Runs and hyperparameters
- Baseline: basis [point, toroidal_eigen_mode_boundary], n_max=12, reg_l1=1e-3, boundary_weight=0.8, per_type_reg {point:2e-3, toroidal_eigen_mode_boundary:8e-4}.
- Hybrid: baseline + two inner_rim_arcs (via basis [point, toroidal_eigen_mode_boundary, inner_rim_arc]), n_max=12, reg_l1=8e-4, boundary_weight=0.8, per_type_reg {point:2e-3, toroidal_eigen_mode_boundary:8e-4, inner_rim_arc:8e-4}. Elements ~4 eigen modes + 2 arcs + 6 points.

Key metrics
- Stage4 (collocation-based, inner-rim belts):
  * Baseline: boundary_mae 8.794e7; offaxis_rel 1.47; belt_rel 0.703; n_images 12.
  * Hybrid arcs: boundary_mae 8.608e7; offaxis_rel 1.18; belt_rel 1.23; n_images 12.
- BEM diagnostics (200x200 r-z grid):
  * Baseline: inner_mean_abs 1.68e8, inner_mean_rel 36.6, max_abs 2.68e10, mean_rel 54.5.
  * Hybrid arcs: inner_mean_abs 5.28e8, inner_mean_rel 73.5, max_abs 6.71e10, mean_rel 59.2.

Interpretation
- For the mid-aspect torus the added inner-rim arcs did not reduce the inner-rim boundary layer; Stage4 belt_rel worsened (0.70?1.23) and BEM diagnostics show higher inner-rim errors. Global metrics remain of similar magnitude. Additional localized primitives or different regularization may be needed; current hybrid is not an improvement over eigen-only.

Files
- spec.json: canonical mid torus spec.
- discovered_system_hybrid.json and discovered_system_baseline.json for the two runs.
- metrics.json: Stage4 + diagnostics summaries.
- Diagnostics NPZ/PNGs: mid_baseline_eigen.*, mid_hybrid_arc.*.

Status
- No convincing improvement from inner-rim arcs in the mid case; treat as negative result. The baseline eigen-only system remains the better option among tested configurations.

Update: ribbon/patch mini-grid (mid only, Stage4 + BEM stability check)
- Configs tried (all n_max=12, two_stage=False):
  - [point, toroidal_eigen_mode_boundary, inner_rim_ribbon] with reg_l1 in {8e-4, 1e-3}, bw=0.8.
  - [point, toroidal_eigen_mode_boundary, inner_rim_ribbon, inner_patch_ring] with reg_l1 in {8e-4, 1e-3}, bw=0.8.
  - [point, toroidal_eigen_mode_boundary, inner_patch_ring] with reg_l1 in {8e-4, 1e-3}, bw=0.75.
- Stage4 results: belt_rel ranged ~0.93–1.70, none better than baseline belt_rel˜0.70; boundary_mae similar to baseline.
- BEM diagnostics (200x200 r–z) for the two “least-bad” candidates:
  - patch-only (reg=1e-3, bw=0.75): inner_mean_rel ˜ 49.8 (worse than baseline 36.6); mean_rel ˜ 77.4; max_rel ˜ 5.2e5.
  - ribbon-only (reg=1e-3, bw=0.8): inner_mean_rel ˜ 67.5; mean_rel ˜ 62.1; max_rel ˜ 4.2e5.
- Conclusion: with current ribbon/patch parametrization and modest regularization, mid-torus hybrids do not improve inner-rim BEM errors and can be less stable (higher max_rel). The eigen-only baseline remains the most robust mid configuration tested so far. Treat ribbon/patch attempts as negative results pending further basis tuning.
