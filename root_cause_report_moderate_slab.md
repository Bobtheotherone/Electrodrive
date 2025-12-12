# Status
- Gate tools now run without exceptions on `runs/three_layer_moderate/v2_intensive_fullfield/` using the canonical spec `specs/planar_three_layer_moderate_h03_eps2_4.json`.
- Verified commands (GPU where applicable): `tools/three_layer_capacity.py`, `tools/check_three_layer_gate1.py`, `tools/images_gate2.py`, `tools/gate3_novelty.py`.
- `discovery_manifest.json` is updated automatically by capacity (conditioning), Gate 1 (metrics + status), Gate 2 (structure status), and Gate 3 (novelty status); no manual edits needed for the pipeline to complete.

# Simple issues fixed
- `tools/three_layer_capacity.py`: added robust spec resolution (absolute or repo-relative) to avoid `FileNotFoundError`; confirmed missing `os` import is present; manifest writing remains intact.
- `tools/check_three_layer_gate1.py`: spec can be omitted (pulls from manifest), manifest auto-updates with `gate1_metrics` and `gate1_status`; added gate thresholds and repo-root path resolution to avoid brittle `runs/specs/...` mistakes.
- `tools/images_gate2.py`: accepts missing `--spec` by resolving from manifest and repo root; uses the same resolver to prevent bad relative paths; continues to write `structure_score`/`gate2_status` even in diagnostic mode.
- `tools/gate3_novelty.py` (new): thin CLI over `update_manifest_with_novelty` with robust spec resolution; prevents the earlier `runs/specs/...` `FileNotFoundError` and fills `novelty_score`/`gate3_status`.

# Root cause analysis

## Ill-conditioning in the moderate slab run
- Capacity with the run geometry/basis (`axis_point,three_layer_images`) produces `cond_est ≈ 1.5e10` (`tools/three_layer_capacity.py --n-points 2048 --ratio-boundary 0.65`), marking `condition_status="ill_conditioned"`.
- SVD on the normalized dense operator (31 columns) shows `s_min ≈ 7.8e-10`, `s_max ≈ 5.13`, `cond ≈ 6.6e9`, i.e., columns are nearly linearly dependent even after normalization.
- Dictionary composition is heavily skewed: 24 `axis_point` vs 2 `slab`, 2 `mirror`, 3 `tail` candidates. Many axis candidates sit far from the interfaces (e.g., z ≈ ±10, ±30, ±50), making their boundary traces highly correlated and driving the near-null singular values.
- Because discovery was run in full-field (no physical subtraction) and with L1 on this ill-conditioned, axis-heavy dictionary, the solver inherits the unstable subspace; conditioning failure is intrinsic to the candidate set, not an isolated numerical glitch.

## Axis-dominated discovered system
- Structural fingerprint of `discovered_system.json` shows `axis_weight_l1_fraction ≈ 0.936` (8 axis images) vs `nonaxis ≈ 0.064` (slab=2, mirror=1, tail=1).
- Non-axis weights are tiny (`three_layer_tail` weight_l1 ≈ 1.1e-4), indicating the solver relied almost entirely on axis columns to fit the field.
- Driver causes: (1) candidate generation floods the dictionary with axis points while offering few slab/mirror/tail options; (2) ill-conditioning plus L1 encourages the optimizer to keep the most correlated (axis) columns and prune weak slab/tail columns; (3) full-field solve magnifies the physical-source imprint, further biasing toward axis images.

## Gate interplay diagnostics
- Gate 1 now records metrics and status; status stays `fail` because `condition_status` is `ill_conditioned` despite small residuals (rel_mae ≈ 4e-6, rel_max ≈ 1.76e-4).
- `images_gate2.py` short-circuits structural scoring when `numeric_status != "ok"` or `condition_status == "ill_conditioned"`, so it writes `gate2_status="n/a"` with a diagnostic note.
- `update_manifest_with_novelty` (via `tools/gate3_novelty.py`) only evaluates novelty when Gate 2 is `pass` or `borderline`; with `gate2_status="n/a"`, it sets `gate3_status="n/a"` and leaves `novelty_score` null. This behavior matches the intended gate ordering (Gate 1 must pass before Gate 2/3 are meaningful).

# Potential future upgrades (not implemented in this RCA)
- Trim and re-balance the three-layer dictionary (reduce far-axis counts, add more slab/mirror/tail coverage) and/or apply per-family scaling before ISTA/LISTA to relieve the near-null subspace.
- Prefer induced-field solves (`--subtract-physical`) or boundary-heavier collocation to reduce dynamic range and axis bias.
- Revisit regularization (group sparsity or LISTA checkpoints) once the dictionary is better conditioned, to curb axis dominance without sacrificing slab support.
