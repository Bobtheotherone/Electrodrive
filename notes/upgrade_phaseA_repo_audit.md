# Phase A – Layered Planar / Three-Layer Audit

## Environment summary
- GPU available: `NVIDIA GeForce RTX 5090 Laptop GPU`; `torch.cuda.is_available() == True`, 1 device. CPU marked weak in AGENTS.
- Constraints: GPU-first for nontrivial numerics; avoid heavy CPU BEM; no external network; prefer `EDE_DEVICE=cuda`, `float32` unless phase demands higher precision.

## Repo layout (high level)
- `electrodrive/` – core library; layered logic in `learn/collocation.py`, `core/planar_stratified_reference.py`, basis/operator/search in `images/`.
- `specs/` – CanonicalSpec JSONs (three-layer slab spec present).
- `experiments/` – experiment plans (three_layer_highcontrast_region1 plan present).
- `runs/` – run outputs and reports (v1 high-contrast run present).
- `docs/research/` – knowledge libraries (`electrostatics_greens_functions_library`, `greens_functions_library`).
- `tests/` – test suite (no layered-planar-specific tests yet).
- `notes/` – audit and other notes (this report).
- `the_vault/` – discovery vault; not used in Phase A.

## AGENTS.md summary
- Role: GPU-first MOI discovery **upgrade** agent (`moi-discovery-upgrade-scientist`).
- Hardware: weak CPU, strong GPU; use GPU for numerics; avoid heavy CPU BEM/runs.
- Network: no external downloads/APIs.
- Lifecycle: Bootstrap → Knowledge/Code Inspection → Design → Execute → Validate → Success/Failure; produce diagnostics/failure reports when gates fail.

## Code-path audit
- `electrodrive/learn/collocation.py`
  - `_solve_analytic` detects three-layer dielectric when dielectrics exist and a point charge in region1; builds `ThreeLayerConfig` and wraps `make_three_layer_solution` (analytic region1-only integral).
  - `_infer_geom_type_from_spec` labels any spec with dielectrics as `"layered_planar"` (no conductor check).
  - `_sample_points_for_spec` layered branch: shrinks x/y (alpha=0.25); z sampling uses layer bounds + charge heights but clamps `z_min` with `max(0.0, min(z_vals))`, biasing to z≥0; boundary sampling for dielectric interfaces only keeps interfaces with `z >= -1e-6` (upper interfaces).
  - Oracle eval: `potential_three_layer_region1` used when analytic meta `kind=planar_three_layer`; otherwise BEM fallback. Domain mask only filters plane/parallel_planes/sphere; layered_planar not masked.
  - Collocation batches cleared oracle cache per call; supports adaptive rounds; subtract_physical flows through discover_images call sites.
- `electrodrive/core/planar_stratified_reference.py`
  - `ThreeLayerConfig` + `potential_three_layer_region1`: Sommerfeld integral for region1 (z ≥ 0) only; reflection coefficients for three-layer; `n_k` default 256, `k_max` heuristic; scales direct term by eps1.
  - `make_three_layer_solution` returns AnalyticSolution-like object with meta `kind=planar_three_layer`; evaluation uses GPU if available. No support for sources in region2/3 or field evaluation below z<0.
- `electrodrive/images/basis.py`
  - Basis registry in `ImageBasisElement.deserialize`; `PointChargeBasis` supports `axis_point` and `three_layer_images` tags but same implementation.
  - `generate_candidate_basis`: layered handling:
    - `wants_three_layer` adds heuristic point charges at fixed depths relative to slab (`-0.25h`, `-h-0.5h`, `-h-1.5h`, `-h-2.5h`, `+0.1h`) using `PointChargeBasis(type_name="three_layer_images")`; no complex depths or interface-aware patterns; ignores permittivity ratios.
    - `axis_point` adds mirrored/perturbed candidates around conductor planes and charges (used here with no conductors).
  - `BasisOperator` defined here (re-exported by `images/operator.py`): matvec/rmatvec, optional row weights, group ids, column-norm estimation; no automatic normalization beyond ISTA call-site scaling.
- `electrodrive/images/operator.py`
  - Thin shim re-exporting `BasisOperator` from basis; no extra conditioning logic.
- `electrodrive/images/search.py`
  - `solve_l1_ista` supports dense or `BasisOperator`; column normalization applied for both modes via estimated norms (but relies on raw basis potentials; large norms still recorded/logged).
  - `discover_images` chooses device/dtype via env; operator mode default True unless env disables. Builds static candidates via `generate_candidate_basis`, supports adaptive collocation, group lasso, LISTA fallback, subtract_physical flag passed to collocation. Logs `basis_operator_stats` when operator mode used; no explicit column renormalization beyond ISTA scaling, no per-family preconditioning.
- `electrodrive/tools/images_discover.py`
  - CLI flags: `--spec`, `--basis`, `--nmax`, `--reg-l1`, `--n-points`, `--ratio-boundary`, `--adaptive-collocation-rounds`, `--restarts`, `--solver`, `--operator-mode`, `--aug-boundary`, `--lambda-group`, `--basis-generator` options, `--subtract-physical`. Wires to `discover_images` with `subtract_physical_potential` passthrough. Operator mode default None→search env logic.

## High-contrast three-layer artifacts
- Spec (`specs/planar_three_layer_eps2_80_sym_h04_region1.json`): **exists**. eps1=eps3=1, eps2=80, slab `z∈[-0.4,0]`, charge q=1 at (0,0,0.2), `BCs="dielectric_interfaces"`, `symmetry=["rot_z"]`, symbols include eps1/2/3, h=0.4, source_region=region1. Matches conceptual geometry; bbox ±1.2 with extended z spans ±10 in dielectrics.
- Experiment plan (`experiments/three_layer_highcontrast_region1/experiment_plan.json`): **exists**. basis `["axis_point","three_layer_images"]`; solver ISTA nmax=12 reg_l1=1e-3 restarts=1 operator_mode=false; collocation n_points=2048 ratio_boundary=0.7 adaptive_rounds=2; device cuda dtype float32; success thresholds boundary_mae<=1e-3, boundary_max<=5e-3, support_size<=8; novelty_expected true; planned_command matches prompt (operator_mode not forced true).
- Run v1 (`runs/three_layer_highcontrast_region1/v1/`): **exists**.
  - `events.jsonl`: operator_mode true (env) despite plan false; adaptive rounds=2; `col_norm_max≈5.45e10`, ISTA non-converged (rel_change ~1e-4) with `bc_norm≈1.7e4`, `int_norm≈1.5e4`.
  - `discovered_system.json`: 12 images, many duplicate `axis_point` at source (z≈0.2) plus spurious far depths (z up to ~99.8, down to ~-20.2) small weights; no structured three-layer pattern.
  - `failure_report.md`: Gates 1–2 failed; attributes ill-conditioning/high contrast, lack of complex slab basis, missing operator normalization focus, interface-focused collocation, subtract-physical option; requests complex image basis + column norming + interface sampling + subtract-physical; notes need for complex-image knowledge libraries.

## Knowledge libraries
- `docs/research/electrostatics_greens_functions_library`: sections include `sec.geometry_three_layer_planar_slab`, `sec.geometry_multi_layer_general_n_layer`, `sec.complex_image_method`, `sec.discrete_complex_images_for_three_layer`, `sec.sommerfeld_integrals`, `sec.reflection_coefficients`; references include layered-media complex-image papers (REF5/6/8/10).
- `docs/research/greens_functions_library`: entries relevant to layered planar/guided modes: `planar_layer_examples.md`, `planar_half_space_single_interface.md`, `planar_spectral_integral_representation.md`, `general_failure_of_finite_images.md`, `general_parameter_scaling_numerical_considerations.md`, `planar_key_references.md`, `spherical_sphere_near_dielectric_interface.md` (hybrid cases). No dedicated complex-pole/thin-slab addenda beyond these; future library may be needed for high-contrast DCIM pole handling.

## Tests overview
- Existing relevant tests: `tests/test_images_discover.py`, `tests/test_images_operator.py`, `tests/test_images_adaptive.py`, `tests/test_collocation_helper.py`, `tests/test_collocation_neural.py`, `tests/test_bem_*`, `tests/test_images_stage0_dense_operator_parity.py`, `tests/test_ista_weight_prior.py`, `tests/test_images_basis_*`. No tests mention layered/three_layer; no collocation vs analytic checks for layered_planar.
- Likely additions: layered collocation sampling distribution vs spec bounds; three-layer analytic oracle parity; operator-mode column-normalization regression for high-contrast layered spec; subtract-physical flag behavior.

## Prioritized TODO for later phases
1. Phase B – three_layer_images basis upgrade (`electrodrive/images/basis.py`): implement structured layered/complex-depth candidate generation tied to slab interfaces/ poles; consider conjugate pairs and interface symmetry.
2. Phase C – ISTA/operator conditioning (`electrodrive/images/search.py`, `images/operator.py`): add robust column normalization/preconditioning for operator mode, log conditioning metrics, guard against 1e10 norms on layered cases.
3. Phase D – layered collocation/oracle (`electrodrive/learn/collocation.py`, `core/planar_stratified_reference.py`): fix z-sampling biases (allow z<0), focus on interfaces, extend analytic reference beyond region1 or add BEM-backed oracle selection; add domain masks for layered_planar.
4. Phase E – subtract-physical integration & rerun (`electrodrive/tools/images_discover.py`, `images/search.py`): ensure layered runs default/optionally subtract physical source; rerun spec with upgraded basis/conditioning; update `runs/three_layer_highcontrast_region1`.
5. Phase F – tests & CI (`tests/`): add layered-planar regression tests (collocation distribution, analytic vs oracle, operator norm bounds, subtract-physical correctness) and CI knobs for GPU-aware small cases.
