Mid-torus axis weight SVD study (2 rings + 4 points, fixed geometry)

Theory inputs (docs/research):
- Functional_Analysis_of_the_Torous.pdf: single-layer boundary operator on the torus is compact, positive, Fredholm with discrete spectrum → Green’s kernel is Hilbert–Schmidt with infinite rank; exact finite-basis representation is ruled out except for trivial geometries. Finite N can only approximate; low effective rank along a restricted parameter set (axis) is plausible.
- toroid.pdf: toroidal harmonics give infinite (m,n) family even for m=0; no truncation expected.
- Inner-Rim Asymptotics and Boundary Layers.pdf: inner-rim boundary layer requires high poloidal content; localized primitives near the inner rim help capture steep gradients.

Experiment design:
- Fixed geometry = mid_bem_highres_trial02 (2 poloidal_ring order=2, radii 1.0573/0.9535, delta_r 0.0833/0.1869; 4 near-surface points).
- Axis grid z = 0.40…0.90 step 0.05 (11 samples). Solve weights only via ISTA (n_colloc=4096, ratio_boundary=0.8, reg_l1=4e-4, point_reg_mult=4, boundary_weight=0.9).
- Dataset: runs/torus/mid_axis_weights.json (weights + Stage metrics).
- SVD of weight matrix W ∈ R^{6×11}: σ/σ1 ≈ [1.0, 0.408, 0.059, 0.0196, 0.0059, 0.0025]; rank thresholds: >1e-1 → 2, >1e-2 → 4.
- Reduced-rank reconstructions (r=2,3) evaluated with high-res BEM (nr=nz=200) at z∈{0.40,0.60,0.70,0.90}.

Key BEM metrics (mean_rel / inner_mean_rel / max_rel):
- Full (r=6): z=0.40 → 8.77 / 10.60 / 7.70e4; z=0.60 → 5.78 / 9.61 / 5.1e3; z=0.70 → 5.68 / 9.37 / 1.50e4; z=0.90 → 19.25 / 10.26 / 4.11e5.
- Rank-2: z=0.40 → 7.86 / 9.46 / 6.82e4; z=0.60 → 5.23 / 8.71 / 4.54e3; z=0.70 → 6.22 / 10.37 / 1.67e4; z=0.90 → 18.46 / 9.73 / 3.92e5.
- Rank-3: z=0.40 → 8.08 / 9.86 / 7.00e4; z=0.60 → 5.19 / 8.59 / 4.50e3; z=0.70 → 6.15 / 10.17 / 1.65e4; z=0.90 → 18.81 / 10.02 / 4.01e5.
Baselines for context: mid_bem_highres_trial02 (z=0.70) mean_rel≈5.39, inner_mean_rel≈7.35, max_rel≈2.25e4.

Interpretation:
- The operator viewpoint and toroidal harmonic theory forbid an exact finite-basis Green’s function; the axis family is at best low-rank. The weight map shows strong decay after two modes (σ2/σ1≈0.41, σ3/σ1≈0.059), consistent with a low Kolmogorov n-width for this 1D manifold.
- Rank-2/3 truncations barely degrade BEM accuracy for z=0.40–0.70 and even reduce inner_mean_rel in some cases (z=0.60). High max_rel persists, especially near z=0.90, indicating unresolved boundary-layer / far-field structure; geometry likely needs an extra localized element for the inner rim or outer belt to tame spikes.
- Conclusion: Evidence supports a very low-rank approximate Green’s function along the axis (effective rank ≈2–3 within 5–10% mean_rel), but theory and high max_rel spikes confirm no exact finite-basis solution; boundary-layer physics keeps the operator infinite-rank.

Artifacts:
- Dataset: mid_axis_weights.json, SVD: mid_axis_weight_svd.json, BEM metrics: metrics_bem.json.
- Systems: full z-sweep, rank-2/3 systems at z=0.60, 0.90; baselines (mid_bem_highres_trial02, mid_baseline_eigen).
- Diagnostics: mid_axis_weight_rank{full,2,3}_z{0.40,0.60,0.70,0.90}.npz.
- Visuals: mid_axis_weight_full_z0.60.png, mid_axis_weight_rank2_z0.60.png, mid_axis_weight_rank2_z0.90.png.
