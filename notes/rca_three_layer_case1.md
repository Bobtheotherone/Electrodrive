# Case 1 – RCA for three-layer slab failures

## 1. Case overview
- Geometry: three-layer slab with eps1=eps3=1, eps2=80, h=0.4, point charge at (0,0,0.2) in region1 (`specs/planar_three_layer_eps2_80_sym_h04_region1.json`). Moderate-contrast diagnostic uses eps2≈4, h≈0.3 (tools/tests defaults).
- Experiments: `experiments/three_layer_highcontrast_region1/experiment_plan.json`; runs `v1` (no subtraction) and `v2_subtractphysical` (free-space subtraction).
- Prior upgrades A–F: geometry-aware three_layer basis motifs, operator-mode column-norm logging/normalization hooks, layered collocation + region1 analytic oracle, subtract-physical path, capacity diagnostics tooling.

## 2. Evidence collected
- High-contrast runs
  - `runs/three_layer_highcontrast_region1/v1`: operator_mode=true, col_norm_min≈4.39e9, col_norm_max≈5.45e10; ISTA non-converged (rel_change≈1.8e-4), bc_norm≈1.75e4, int_norm≈1.50e4. Support: 12 images (11 axis_point, 1 three_layer_images) with many duplicates at z=0.2 and spurious far depths (z≈79–100, z≈-20).
  - `runs/three_layer_highcontrast_region1/v2_subtractphysical`: operator_mode=true, col_norm_min≈3.67e9, col_norm_max≈4.28e10; ISTA non-converged (rel_change≈1.8e-4), bc_norm≈4.29e5, int_norm≈2.64e5. Support: 12 images (8 axis_point, 4 three_layer_images) clustered near z≈0.2/-0.1/−0.2/−0.3 with tiny tails at z≈-20, z≈99.8; still no structured slab ladder.
- Gate 1 checker (this run)
  - v2_subtractphysical, subtract-physical induced targets, 512 pts, ratio_boundary=0.6: MAE≈3.0e3, max≈8.55e4, boundary MAE≈2.61e3 (thresholds 1e-3 / 5e-3).
  - v1, full field, 512 pts, ratio_boundary=0.6: MAE≈4.34e2, max≈6.61e3, boundary MAE≈2.02e2.
- Dense LS capacity (tools/three_layer_capacity.py, device=cuda, dtype=float32, n_points=512, ratio_boundary=0.6)
  - Moderate contrast (eps2=4, h=0.3): base MAE≈6.52e12, max≈1.41e14; boundary MAE≈5.84e12. Column norms 2.56e9–4.40e10. Family scaling improved fit but still huge (scaled MAE≈1.30e8, max≈2.08e9) with norms 5.33e9–9.04e10; median-based family scales ~2.08 (axis_point), ~0.78–1.08 (three_layer_*).
  - High contrast (spec eps2=80, h=0.4): base MAE≈1.34e13, max≈5.63e14; boundary MAE≈1.47e13. Column norms 1.28e9–2.26e10. Family scaling made it worse here (scaled MAE≈2.74e11, max≈2.20e12) despite similar scales (~2.22 for axis_point, ~0.83–1.07 for three_layer_*).
  - Column norms stay O(1e9–1e10) regardless of scaling; LS residuals remain orders of magnitude above targets, confirming capacity/conditioning issues even before sparsity.
- Knowledge-library notes
  - `electrostatics_greens_functions_library`: three-layer slabs require Sommerfeld integrals with reflection spectrum containing branch cut plus guided-mode pole; DCIM typically separates the pole and fits remaining spectrum with complex depths. Complex image method places sources at complex depths; near-interface fields are hardest to approximate and often need many terms.
  - `greens_functions_library/library/entries/general_failure_of_finite_images.md`: finite discrete images generally fail for dielectric slabs; continuous image distributions or infinite/complex series are needed except in special geometries (plane, sphere, etc.).

## 3. Hypothesis analysis (H1–H5)
- **H1 – Basis expressiveness/capacity**: Strongly supported. Dense LS fails by 8–14 orders of magnitude for moderate and high contrast; discovered systems collapse to duplicated real charges and sparse shallow images with no slab-tail ladder. Knowledge base notes guided-mode poles and need for complex/continuous images, incompatible with the current shallow real-depth set.
- **H2 – Dictionary scaling/conditioning**: Supported as contributing factor. Column norms are O(1e9–1e10) in both runs and LS dictionaries; ISTA normalization cannot tame them, and per-family scaling either barely helps (moderate case) or worsens (high contrast). Indicates missing amplitude normalization tied to reflection coefficients/complex decay.
- **H3 – Collocation/oracle mismatch**: Weak/partial support. Collocation uses region1 analytic oracle and interface-focused sampling; Gate 1 uses consistent induced/full targets. While near-source samples may inflate magnitude, the failure persists across induced vs full targets and LS solves, so mismatch is not primary.
- **H4 – Subtract-physical semantics**: Partially supported. Subtracting free-space physical field did not reduce norms or errors (Gate 1 still O(1e3–1e4); column norms unchanged). Subtraction is too weak because it omits layered reflections; dynamic range remains dominated by induced part.
- **H5 – Implementation bugs vs structural limits**: Largely refuted for major bugs. The tooling, oracles, and subtract-physical wiring behave as expected; failures align with known nonexistence of finite real images and missing complex-depth amplitudes rather than coding errors.

## 4. Root cause conclusion
- Primary: The current real-depth image basis (axis_point + three_layer_images motifs) lacks the expressiveness to approximate the three-layer Sommerfeld response, which includes guided-mode poles and branch cuts; finite real images cannot capture these features, leading to huge LS residuals even before sparsity. This is exacerbated by absent amplitude normalization tied to layered reflection coefficients, yielding intrinsically ill-scaled columns (O(1e10)) that stall ISTA.
- Secondary: Free-space subtract-physical only removes the direct charge, leaving the high-contrast induced field untouched; operator-mode scaling and interface-focused collocation cannot compensate for the ill-conditioned dictionary.

## 5. Recommended directions (not implemented here)
- Basis redesign: Introduce complex/decaing depth images (DCIM-style) with explicit handling of the slab guided-mode pole and amplitude factors from reflection coefficients; include slab-tail and interior motifs with complex-conjugate pairs.
- Dictionary conditioning: Add optional explicit column normalization or family-wise preconditioning in `BasisOperator`/search paths; consider normalizing against analytic layered reference scales rather than raw Coulomb magnitudes.
- Improved subtraction: For layered cases, subtract an analytic layered reference (region1 Sommerfeld direct+reflected) instead of free-space only to reduce target dynamic range before solving.
- Collocation/diagnostics: Add regression diagnostics on moderate-contrast slab to ensure analytic vs BEM agreement and bounded column norms; bias sampling toward interfaces and guided-mode decay lengths with guardrails against near-source singular spikes.
- Knowledge extensions: Capture high-contrast three-layer DCIM recipes (pole isolation, rational fits of Reff(k)) in the library to guide depth/amplitude initialization for future phases.
