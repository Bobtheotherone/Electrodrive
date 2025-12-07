Random torus explorer (thin + mid) with novelty detection

Setup
- Random Stage4 sweeps (10 trials each) over torus_thin and torus_mid using tools/random_torus_explorer.py.
- Basis pool sampled per run: point + >=1 global (poloidal_ring, ring_ladder_inner, toroidal_mode_cluster, toroidal_eigen_mode_boundary/offaxis) with optional inner_rim_arc/ribbon/patch; n_max in {4..24}, reg_l1 in {3e-4..1e-2}, bw in {0.5,0.7,0.8,0.9}, two_stage in {False,True}, restarts<=1. Per-type reg scaled randomly (points heavier, locals 0.5–1.5x reg).
- Belts: inner-focused (R-a to R+0.5a, z in {0, ±0.2a}).

Runs
- Total Stage4 trials: 20 (10 thin, 10 mid). Logged to runs/torus/stage4_metrics_random_sweep.json.
- Interesting (collocation) per spec:
  * thin: best boundary_mae ˜ 8.66e7 (n=8) but BEM inner errors worsened vs baseline ? negative.
  * mid: several low boundary_mae runs; candidate seed5500358 (basis [point, toroidal_eigen_mode_boundary, poloidal_ring, inner_rim_arc, inner_rim_ribbon], n_max=6, reg_l1=3e-4, bw=0.9, two_stage=True, restarts=1) looked promising.

BEM diagnostics (200x200 r–z)
- Baseline mid eigen-only (from prior work): inner_mean_rel ˜ 36.6, mean_rel ˜ 54.5, max_rel ˜ 3.5e5.
- Candidate seed5500358: inner_mean_rel ˜ 26.9, mean_rel ˜ 31.4, max_rel ˜ 2.4e5 (improves inner and global rel errors with only 6 images). Stage4: boundary_mae ˜ 5.09e7, belt_rel ˜ 0.79.
- Thin candidate seed1099217: inner_mean_rel ˜ 12.5 vs baseline 2.25 (worse) ? discard.

Artifacts
- specs: spec_mid.json (canonical mid torus).
- discovered systems: discovered_mid_candidate.json (seed5500358), discovered_mid_baseline.json (baseline eigen-only).
- metrics.json: Stage4 + BEM stats for baseline vs candidate.
- Diagnostics: mid_random_seed5500358.*, mid_random_reg6e4.* (same config retested), mid_baseline_eigen.*.

Conclusion (numerical conjecture)
- Random exploration found a mid-torus hybrid (6 images: ~2 poloidal rings + 4 points after sparsity) combining global modes and a light inner_rim_arc/ribbon presence that lowers inner-rim and overall BEM relative errors versus the eigen-only baseline while keeping n_images small.
- Thin-side random candidates were not better than baseline. Further mid refinements could try nearby regs/boundary weights, but current candidate already improves inner metrics without instability.
