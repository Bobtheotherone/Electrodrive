# CHANGELOG_AI_UPGRADE

## 2025-12-15
- Added GFDSL foundation (AST, param system, schema-aware IO, canonicalization/hashing, validation stubs) per steps 4.0–4.4; no integration with existing pipeline yet.
- Tests: `python -m pytest electrodrive/gfdsl/tests -q --maxfail=1`.
- Notes: Evaluation kernels and adapters remain TODO; GFDSL stays opt-in and isolated.
- GFDSL Param no longer pins tensors to CPU; GPU-first materialization via `Param.value(device=…)` added; hashing/serialization remain CPU-safe cold paths.

## 2025-12-16
- Implemented compiler lowering contract (CompileContext, CoeffSlot, LinearContribution, ColumnEvaluator variants) and evaluators for RealImageCharge, Dipole, and ConjugatePair(ComplexImageCharge) with CUDA-first execution.
- Added targeted tests for real/complex primitives, dense/operator parity, and autograd gradients.
- Tests: `python -m pytest electrodrive/gfdsl/tests -q --maxfail=1`.
- Notes: ComplexImageCharge lowers through ConjugatePair only; operator evaluators retain dense debug paths for parity checks.

## 2025-12-17
- Added GFDSL motif macros (MirrorAcrossPlane, ImageLadder) with grouping metadata propagation, legacy adapter to ImageBasisElement-like wrappers, and layered-media node stubs (InterfacePole, BranchCutApprox, DCIMBlock) with placeholder evaluators.
- Expanded tests for macros, adapter caching, placeholder stubs, and operator parity including macros; updated docs with JSON example and adapter usage.
- Tests: `python -m pytest electrodrive/gfdsl/tests -q --maxfail=1`.
- Notes: Layered evaluators remain TODO and raise a clear NotImplementedError; existing pipeline behavior unchanged without opt-in adapter usage.

## 2025-12-18
- Enforced coefficient-only amplitudes for RealImageCharge/Dipole (new FixedCharge/FixedDipole for known sources), enabled Sum-aware ladder/mirror macros, and implemented streaming operator matvec/rmatvec kernels with no eval_columns fallback.
- Added operator no-dense-fallback tests, updated legacy adapter expectations, and refreshed README example/JSON for the new semantics.
- Tests: `python -m pytest electrodrive/gfdsl/tests -q --maxfail=1`.
- Notes: Layered evaluators still placeholders; unknown amplitudes must be provided via solver coefficients.
- Registered fixed-source nodes in the schema/registry, enforced separate fixed_term convention across Sum/Mirror/Ladder, and added anti-double-count regression tests.
