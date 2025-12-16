import torch

from electrodrive.gfdsl.ast import DipoleNode, Param, RealImageChargeNode, SumNode
from electrodrive.gfdsl.compile import CompileContext, linear_contribution_to_legacy_basis


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_legacy_adapter_columns_and_cache():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    real_node = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, 0.2], device=device)),
        }
    )
    dipole_node = DipoleNode(
        params={
            "position": Param(torch.tensor([0.1, -0.1, 0.3], device=device)),
        }
    )
    program = SumNode(children=(real_node, dipole_node))
    contrib = program.lower(ctx)

    orig_eval = contrib.evaluator.eval_columns
    call_counter = {"count": 0}

    def counting_eval(X: torch.Tensor) -> torch.Tensor:
        call_counter["count"] += 1
        return orig_eval(X)

    contrib.evaluator.eval_columns = counting_eval  # type: ignore[assignment]

    wrappers = linear_contribution_to_legacy_basis(contrib)
    X = torch.randn(6, 3, device=device, dtype=ctx.dtype)
    Phi_expected = orig_eval(X)

    outputs = [w.potential(X) for w in wrappers]

    assert call_counter["count"] == 1  # cached columns reused across wrappers
    for k, out in enumerate(outputs):
        assert torch.allclose(out, Phi_expected[:, k], rtol=1e-5, atol=1e-6)
        assert out.device.type == ctx.device.type
    assert wrappers[0].type == "gfdsl_column"
    assert "slot_id" in wrappers[0].params
    assert isinstance(wrappers[0].description(), str)
