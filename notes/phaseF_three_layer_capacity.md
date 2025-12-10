# Phase F – Three-layer Capacity & Evaluation Diagnostics

## Gate 1 semantics (subtract-physical)
- Added `tools/check_three_layer_gate1.py` to evaluate Gate 1 with subtract-physical awareness (uses induced targets when subtracting physical).
- Corrected Gate 1 for v2_subtractphysical (induced comparison, 512 pts, ratio_boundary=0.6):  
  - MAE ≈ 3.1e3, Max ≈ 8.55e4, Boundary MAE ≈ 2.6e3, Boundary Max ≈ 8.55e4.  
  - Errors remain far above thresholds (1e-3/5e-3).

## Dense LS capacity (tools/three_layer_capacity.py, normalized dictionary)
- Method: build collocation (full field), dense dictionary from candidate basis (axis_point + three_layer_images, 32 candidates), normalize columns via `BasisOperator.normalized_dense`, solve torch lstsq; optional per-family scaling.
- Thresholds (diagnostic, relative errors):  
  - Basis-capable: `rel_mae <= 1e-2` and `rel_boundary_mae <= 1e-2`.  
  - Basis-stressed: `1e-2 < rel_mae <= 5e-2` (or boundary analog).  
  - Basis-incapable: `rel_mae > 5e-2` or `rel_boundary_mae > 5e-2`, or cond_est >> 1e3.
- Moderate contrast (eps2=4.0, h=0.3, n_points=256, ratio_boundary=0.6, seed=0, normalized):
  - rel_mae ≈ 7.74e-3, rel_boundary_mae ≈ 7.48e-3 → **basis-capable** by rel thresholds.
  - col_norm_min/max ≈ 1.0/1.0 (normalized), cond_est ≈ 3.71e10 (ill-conditioned but rel error acceptable).
- High contrast (specs/planar_three_layer_eps2_80_sym_h04_region1.json, n_points=256, ratio_boundary=0.6, seed=1, normalized):
  - rel_mae ≈ 2.73e-2, rel_boundary_mae ≈ 2.68e-2 → **basis-stressed** (fails capable threshold).
  - col_norm_min/max ≈ 1.0/1.0 (normalized), cond_est ≈ 5.66e10 (severely ill-conditioned).
- Interpretation: Normalization tames column norms to O(1), but conditioning remains poor. Moderate contrast meets rel error targets despite huge cond_est (suggests spectrum dominated but still solvable). High contrast sits in basis-stressed; large cond_est and higher rel errors indicate structural limits of current real-depth basis.

## Recommendations
1) Basis redesign: Introduce complex/decaying depth basis with explicit amplitude scaling tied to reflection coefficients; current real-only depths struggle on high contrast.
2) Maintain layered/homogeneous-eps1 subtraction; continue using normalized dictionaries for diagnostics.
3) Capacity gate: classify moderate as capable, high contrast as stressed/incapable; avoid GPU discovery on stressed/incapable regimes until basis is upgraded.
