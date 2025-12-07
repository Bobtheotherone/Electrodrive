Mid-torus axis sweep for the fixed 2-ring + 4-point geometry (mid_bem_highres_trial02).

Setup
- Spec: specs/torus_axis_point_mid.json (R=1.0, a=0.35), charge moved along axis z∈{0.40,0.50,0.60,0.70,0.80,0.90}.
- Geometry fixed to mid_bem_highres_trial02: rings [(r=1.0573, Δr=0.0833, order=2), (r=0.9535, Δr=0.1869, order=2)] and points [(-0.0769,0.8454,0.0114), (-0.1260,0.8981,0.0073), (-0.8416,0.0698,-0.0417), (-0.8496,0.0097,0.0897)] in (x,y,z).
- Solve per z: ISTA on fixed geometry, n_colloc=3072, ratio_boundary=0.8, reg_l1=4e-4, point_reg_mult=4, boundary_weight=0.9.
- Diagnostics: high-res BEM grid 220×220 on r∈[R−1.5a,R+1.5a], z∈[−1.5a,1.5a]; BEM cfg use_gpu fp64, max_refine=5, near_quad eval on.
- FMM sanity: bem_matvec_gpu backend="external" with make_laplace_fmm_backend vs torch_tiled reference.

Results (BEM grid metrics)
- z=0.40 mean_rel=7.49, inner_mean_rel=11.41, max_rel=2.8e4
- z=0.50 mean_rel=7.29, inner_mean_rel=17.78, max_rel=3.18e4
- z=0.60 mean_rel=7.47, inner_mean_rel=14.53, max_rel=2.08e4
- z=0.70 mean_rel=5.51, inner_mean_rel=7.37, max_rel=2.31e4
- z=0.80 mean_rel=10.30, inner_mean_rel=13.21, max_rel=7.60e4
- z=0.90 mean_rel=7.14, inner_mean_rel=14.61, max_rel=2.42e4
- Baseline (trial02 at z=0.70) from mid_bem_highres_trial02.npz: mean_rel≈5.39, inner_mean_rel≈7.35, max_rel≈2.25e4.

Observations
- Geometry is most stable near z≈0.70 (comparable to trial02), with noticeable degradation at z=0.80 (mean_rel>10, high max_rel) and consistently high max_rel across the sweep.
- Inner-rim error climbs as the source moves away from 0.70; belts stay O(10–18%) except z=0.40 outer band spike.
- Stage-4 collocation metrics remain large (offaxis_rel >1) indicating residual mismatch despite fixed geometry.

FMM check
- bem_matvec_gpu torch_tiled vs external FMM on the z=0.70 BEM system: rel_l2_err ≈ 1.24e-2 over 1344 panels (FMM backend creation succeeded).

Artifacts
- Metrics: metrics_stage4.json, metrics_bem.json.
- Systems: systems/mid_axis_sweep_z*.json plus original/mid_bem_highres_trial02.json and mid_baseline_eigen.json.
- Diagnostics: diagnostics/mid_axis_sweep_z*.npz, mid_bem_highres_trial02.npz.
- Visuals: visuals/mid_axis_sweep_z0.70.png (best) and z0.80.png (degraded).

Next steps
- Try adaptive boundary weighting or higher n_colloc to see if weights tighten inner rim errors without altering geometry.
- Explore slight z-dependent scaling of ring radii/delta_r while keeping point set fixed to probe robustness.
- Consider per-z or piecewise-linear weight interpolation to quantify how close this geometry is to a finite-basis Green’s function along the axis.
