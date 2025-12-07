Mid-torus local geometry explorer + high-res BEM refinement (numerical conjecture)

Geometry
- Spec: specs/torus_axis_point_mid.json (R=1.0, a=0.35), grounded torus; point charge q=1 at (0,0,0.7).

Earlier context (kept for completeness)
- Baseline eigen-only (12 images): mean_rel ˜ 54.47, inner_mean_rel ˜ 36.61, max_rel ˜ 3.53e5 (BEM 200x200).
- Seed S0 (2 poloidal rings + 4 points, seed5500358): mean_rel ˜ 31.41, inner_mean_rel ˜ 26.93, max_rel ˜ 2.43e5.
- Trial003 (previous best local, 2 rings + 4 points): mean_rel ˜ 19.32, inner_mean_rel ˜ 18.18, max_rel ˜ 1.55e5.
- Ribbon/patch mini-grids were negative for mid torus (no improvement over baseline).

High-accuracy BEM refinement (this run)
- Implemented tools/mid_torus_bem_fmm_refine.py with high-res BEM cfg: fp64, max_refine_passes=5, min_refine_passes=2, gmres_tol=1e-8, near_quadrature=True, tile_mem_divisor=2.5, target_vram_fraction=0.9.
- 12 perturbation trials around the seed geometry (jitter ~0.02 on ring radii/delta_r and point ?/f/z, occasional extra point; reg_l1 in {2e-4..6e-4}, boundary_weight in {0.9,0.95}).
- Best BEM candidate: mid_bem_highres_trial02 (still 2 rings + 4 points, n_images=6).
  - Ring params: radius˜1.057 (order 2, ?˜0.083), radius˜0.953 (order 2, ?˜0.187).
  - Point params: ?˜0.85–0.91, |z|?0.09, f near ~95–180°.
  - High-res BEM (220x220 r–z): mean_rel ˜ 5.39, inner_mean_rel ˜ 7.35, max_rel ˜ 2.25e4. (Inner_mean_abs ˜ 3.89e7.)
  - This surpasses trial003 and seed by a wide margin while keeping 6 images.
- All BEM trial metrics logged to runs/torus/stage4_metrics_mid_bem_highres_local.json; diagnostics NPZ+PNGs stored per trial.

FMM attempt
- get_oracle_solution(mode="fmm") is unavailable in this codebase; attempts returned "FMM oracle unavailable" for baseline/seed/trial003/best_bem. No FMM metrics could be computed. Ranking is therefore BEM-only. If FMM support appears later, rerun diagnostics on the same systems.

Artifacts (copied here)
- Discovered systems: discovered_mid_baseline.json, discovered_mid_seed.json, discovered_mid_trial003.json, discovered_mid_bem_highres_trial02.json.
- Diagnostics: mid_baseline_eigen.*, mid_random_seed5500358.*, mid_local_seed5500358_trial003.*, mid_bem_highres_trial02.* (NPZ + PNGs).
- Metrics: stage4_metrics_mid_bem_highres_local.json, mid_candidate_metrics_combined.json.
- Visualization: mid_bem_highres_trial02.png (torus + images + r–z equipotentials).

Conclusion (numerical conjecture)
- A 6-image system (two poloidal rings, both order 2, plus four near-surface points) achieves BEM mean_rel ˜ 5.4 and inner_mean_rel ˜ 7.4 on the mid torus with an on-axis charge at z=0.7, greatly improving over the 12-image eigen-only baseline. This is based on high-res BEM; FMM validation is pending due to unavailable oracle. Treat as a numerical conjecture; further confirmation with FMM or denser grids is encouraged.
