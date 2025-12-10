# Three-Layer Basis Design, Capacity, and Future DCIM Direction (ECO-LP-001)

## Scope and status
- **ECO-LP-001** targets layered planar / three-layer slabs: improve diagnostics, normalization, subtract-physical, and capacity clarity.
- Phases A–E completed:
  - Phase B: relative metrics added to `tools/check_three_layer_gate1.py` and `tools/three_layer_capacity.py`.
  - Phase C: diagnostic dictionary normalization via `BasisOperator.normalized_dense`, used in capacity tool.
  - Phase D: layered subtract-physical now subtracts a homogeneous-eps1 reference; tests extended for layered reconstruction.
  - Phase E: capacity diagnostics classify moderate vs high contrast via relative LS errors and cond_est; notes updated.
- This doc captures acceptance criteria, risks, and future basis work (DCIM-style `three_layer_dcim`) as a plan—no code changes in Phase F.

## Current behavior and acceptance criteria
- **Metrics/diagnostics**: Gate 1 and capacity tools emit absolute and relative errors. Capacity tool reports normalized col norms and `cond_est`.
- **Normalization (diagnostics-only)**: `BasisOperator.normalized_dense` delivers O(1) column norms for LS analysis; not used in discovery yet.
- **Layered subtract-physical**: `get_collocation_data(..., subtract_physical_potential=True)` uses homogeneous-eps1 reference for layered planar; plane path unchanged. Reconstruction test passes with rel error < 1e-2.
- **Capacity clarity**: Relative LS metrics used to classify regimes:
  - Capable: `rel_mae <= 1e-2` and `rel_boundary_mae <= 1e-2`.
  - Stressed: `1e-2 < rel_mae <= 5e-2` (or boundary analog).
  - Incapable: `rel_mae > 5e-2` or `rel_boundary_mae > 5e-2`, or `cond_est >> 1e3`.
- **Measured regimes (normalized LS, n_points~256)**:
  - Moderate contrast (eps2=4.0, h=0.3): `rel_mae ~7.7e-3`, `rel_b ~7.5e-3`, `col_norms ~1`, `cond_est ~3.7e10` ⇒ **basis-capable** (despite ill-conditioning).
  - High contrast (eps2=80, h=0.4): `rel_mae ~2.7e-2`, `rel_b ~2.7e-2`, `col_norms ~1`, `cond_est ~5.7e10` ⇒ **basis-stressed**; large cond_est suggests structural limits with current real-depth basis.
- **Acceptance**: ECO-LP-001 considered complete with metrics, normalization diagnostics, layered subtraction, capacity classification documented and tests passing; DCIM basis and capacity gate wiring left for future ECO.

## Risks and structural limits
- High cond_est (O(1e10)) even after normalization implies fragile solves; discovery runs may remain unstable in stressed/incapable regimes.
- Layered subtract-physical remains opt-in; enabling it changes numerical behavior (reduced magnitudes) but keeps backward compatibility off by default.
- DCIM-style basis will add complexity; must be carefully gated and tested. Even with DCIM, extreme contrasts may remain basis-incapable for small finite dictionaries.

## Future work: DCIM-style basis (`three_layer_dcim`) – design sketch
- **Activation**: Only when `basis_types` includes `"three_layer_dcim"` and `BCs=="dielectric_interfaces"` with exactly 3 dielectric layers forming a z-stack.
- **Depth structure (real doublets approximating complex poles)**:
  - Mirror depth: `z_m = 2 * top_interface - z0`.
  - Decay-length motifs tied to slab thickness/contrast: choose effective `ℓ_i ~ {0.25h, 0.5h, 1.0h}` or contrast-informed scales; place depths `z = bottom_interface - ℓ_i` and `z = bottom_interface - 3ℓ_i`.
  - Optional additional tails further below slab to mimic guided decay; keep small fixed set to avoid explosion.
  - Grouping: `family_name="three_layer_dcim"`, `motif_index` per decay tier; explicit grouping supports diagnostics and potential per-family scaling.
- **Implementation targets (future ECO)**:
  - Add generator branch in `electrodrive/images/basis.py` for `"three_layer_dcim"`; ensure determinism and gating.
  - Tests in `tests/test_images_three_layer_dcim_basis.py`:
    - Activates only for valid three-layer specs; empty otherwise.
    - Depth set matches design (mirror + decay-length tiers).
    - Deterministic ordering.
  - Optional design validation: capacity tool run with `"three_layer_images,three_layer_dcim"` to compare rel errors on stressed/high-contrast cases.

## Capacity gate wiring (future ECO)
- Add a discovery-time flag (e.g., `--capacity-gate`) to skip runs when diagnostics classify a spec as stressed/incapable based on rel metrics and cond_est.
- Gate would:
  - Run a small normalized LS capacity check.
  - Classify using thresholds above.
  - Abort discovery or downgrade priority for stressed/incapable regimes.
- Not part of ECO-LP-001 implementation; to be planned with DCIM rollout.

## Implementation sequencing (completed vs future)
- Completed (ECO-LP-001 Phases B–E): relative metrics, normalized diagnostics, layered subtract-physical, capacity classifications.
- Pending (future ECO): introduce `three_layer_dcim`, tune decay-lengths, wire capacity gate into discovery CLI, and add regression tests for DCIM activation and capacity gating.
