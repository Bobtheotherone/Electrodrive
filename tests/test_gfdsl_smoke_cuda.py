import pytest
import torch

from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    Param,
    RealImageChargeNode,
    SoftplusTransform,
    SumNode,
)
from electrodrive.gfdsl.compile import CompileContext, lower_program, validate_program
from electrodrive.gfdsl.io import deserialize_program, serialize_program


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GFDSL smoke test"
)


def test_gfdsl_smoke_cuda_roundtrip_and_eval():
    device = torch.device("cuda")
    charge = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.1, -0.2, 0.3], device=device)),
        }
    )
    complex_child = ComplexImageChargeNode(
        params={
            "x": Param(torch.tensor(0.0, device=device)),
            "y": Param(torch.tensor(0.0, device=device)),
            "a": Param(torch.tensor(0.5, device=device)),
            "b": Param(raw=torch.tensor(0.25, device=device), transform=SoftplusTransform(min=1e-3)),
        }
    )
    pair = ConjugatePairNode(children=(complex_child,))
    program = SumNode(children=(charge, pair))

    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")
    validate_program(program, ctx)

    payload = serialize_program(program)
    roundtrip = deserialize_program(payload)
    assert program.canonical_dict(include_raw=True) == roundtrip.canonical_dict(include_raw=True)
    assert program.full_hash() == roundtrip.full_hash()

    contrib = lower_program(roundtrip, ctx)
    X = torch.randn(8, 3, device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)
    assert Phi.is_cuda
    assert Phi.shape == (8, contrib.evaluator.K)
    assert torch.isfinite(Phi).all()

    w = torch.randn(contrib.evaluator.K, device=device, dtype=ctx.dtype)
    V = contrib.evaluator.matvec(w, X)
    assert V.is_cuda
    assert V.shape == (8,)
    assert torch.isfinite(V).all()
