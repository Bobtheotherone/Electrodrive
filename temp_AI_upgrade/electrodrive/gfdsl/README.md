# GFDSL (Green's Function DSL)

Experimental, opt-in package for expressing Green's function programs as typed ASTs. The package is **not integrated into the default discovery pipeline**; existing behavior remains unchanged until explicit adapters are introduced.

## What exists now
- AST node definitions and registry.
- Parameter/transform system with JSON serialization.
- Canonicalization + stable hashing (structure and value-aware).
- Schema-aware IO with forward-compatible `OpaqueNode` handling.
- Validation helpers for early error detection.
- Lowering contract (`CompileContext`, `CoeffSlot`, `LinearContribution`) with dense + operator evaluators.
- Differentiable kernels for `RealImageCharge`, `Dipole` (vector coefficient mode), and `ConjugatePair(ComplexImageCharge)` that default to CUDA when available.
- Motif macros (`MirrorAcrossPlane`, `ImageLadder`) with grouping metadata propagation and compositional lowering; operator backend avoids dense fallback.
- Legacy adapter that exposes GFDSL columns as ImageBasisElement-like wrappers without altering the existing discovery pipeline.
- Fixed-source primitives (`FixedChargeNode`, `FixedDipoleNode`) for known amplitudes; unknown amplitudes always reside in solver coefficient slots.
- Layered-media nodes (`InterfacePole`, `BranchCutApprox`, `DCIMBlock`) are present with validation/serialization; evaluators are stubs that raise a clear placeholder message until implemented.

## Example
```python
from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    RealImageChargeNode,
    SumNode,
    Param,
    SoftplusTransform,
)
import torch
from electrodrive.gfdsl.io import serialize_program_json, deserialize_program
from electrodrive.gfdsl.compile import (
    CompileContext,
    validate_program,
    linear_contribution_to_legacy_basis,
)

program = SumNode(
    children=(
        MirrorAcrossPlaneNode(
            children=(
                RealImageChargeNode(
                    params={
                        "position": Param([0.0, 0.0, 0.8]),
                    }
                ),
            ),
            params={"z0": Param(0.0)},
            meta={"sign": -1, "group_policy": "override"},
        ),
        ConjugatePairNode(
            children=(
                ComplexImageChargeNode(
                    params={
                        "x": Param(0.1),
                        "y": Param(-0.2),
                        "a": Param(0.4),
                        "b": Param(raw=[0.0, 0.7], transform=SoftplusTransform(min=1e-3)),
                    }
                ),
            )
        ),
        RealImageChargeNode(
            params={
                "position": Param([0.0, 0.0, 1.0]),
                "charge": Param(1.0),
            }
        ),
    )
)

validate_program(program)
json_payload = serialize_program_json(program)
roundtrip_program = deserialize_program(json_payload)
assert program.canonical_dict(include_raw=True) == roundtrip_program.canonical_dict(include_raw=True)

# Compile + evaluate on GPU if available (CPU fallback otherwise)
ctx = CompileContext()  # defaults to cuda when torch.cuda.is_available()
contrib = program.lower(ctx)
targets = torch.tensor([[0.1, 0.0, 0.5]], device=ctx.device)
Phi = contrib.evaluator.eval_columns(targets)
w = torch.tensor([1.0, 0.5, -0.2], device=ctx.device)  # [mirror real charge, complex pair (re, im)]
potentials = contrib.evaluator.matvec(w, targets) + (contrib.fixed_term(targets) if contrib.fixed_term else 0.0)
assert potentials.device.type == ("cuda" if torch.cuda.is_available() else "cpu")

# Legacy adapter (opt-in): wrap columns as ImageBasisElement-like objects
legacy_basis = linear_contribution_to_legacy_basis(contrib)
legacy_sum = sum(elem.potential(targets) for elem in legacy_basis)
assert torch.allclose(potentials, legacy_sum, rtol=1e-5, atol=1e-6)
```

### Minimal JSON payload
```json
{
  "schema_name": "electrodrive.gfdsl",
  "schema_version": 1,
  "program": {
    "node_type": "mirror_across_plane",
    "params": {"z0": {"raw": 0.0, "transform": {"type": "identity"}, "trainable": true, "dtype_policy": "work"}},
    "meta": {"sign": -1},
    "children": [
      {
        "node_type": "real_image_charge",
        "params": {
          "position": {"raw": [0.0, 0.0, 0.8], "transform": {"type": "identity"}, "trainable": true, "dtype_policy": "work"}
        },
        "children": [],
        "meta": {}
      }
    ]
  }
}
```

## Notes
- Canonicalization sorts child structures for stable hashing; use `structure_hash()` or `full_hash()` on any node.
- Unknown node types deserialize to `OpaqueNode` so newer producers stay forward compatible.
- Lowering supports dense dictionary materialization and operator parity (`matvec`/`rmatvec`).
- Existing discovery pipeline remains untouched; GFDSL is opt-in and isolated until adapters are added.
- Layered-media evaluators are placeholders and currently raise `NotImplementedError` with a clear message until implemented in `electrodrive/gfdsl/eval/layered.py`.
- Unknown amplitudes live in solver coefficient slots; use `FixedChargeNode`/`FixedDipoleNode` for fixed sources to avoid double-scaling.
- By convention, `matvec` computes only coefficient-weighted columns; add `fixed_term(X)` separately when present.
