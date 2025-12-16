import torch

from electrodrive.gfdsl.ast import DipoleNode, Param, RealImageChargeNode
from electrodrive.gfdsl.compile import CompileContext
from electrodrive.utils.config import K_E


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_real_image_charge_matches_formula():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    position = torch.tensor([0.1, -0.2, 0.5], device=device)
    node = RealImageChargeNode(
        params={
            "position": Param(position),
        }
    )

    X = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.7], [-0.2, 0.3, 0.4]],
        device=device,
        dtype=ctx.dtype,
    )
    contrib = node.lower(ctx)
    Phi = contrib.evaluator.eval_columns(X)

    r = torch.linalg.norm(X - position, dim=1).clamp_min(1e-9)
    expected = (float(K_E)) / r
    assert torch.allclose(Phi[:, 0], expected, rtol=1e-5, atol=1e-6)
    # Weighted potential with coefficient q
    q = torch.tensor([1.25], device=device, dtype=ctx.dtype)
    pot = contrib.evaluator.matvec(q, X)
    assert torch.allclose(pot, expected * q[0], rtol=1e-5, atol=1e-6)
    assert Phi.device.type == ctx.device.type


def test_dipole_matches_closed_form():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    position = torch.tensor([0.05, 0.1, -0.2], device=device)
    node = DipoleNode(
        params={
            "position": Param(position),
        }
    )

    X = torch.tensor(
        [[0.3, -0.1, 0.0], [-0.2, 0.4, 0.5], [0.0, 0.0, 0.2]],
        device=device,
        dtype=ctx.dtype,
    )
    contrib = node.lower(ctx)
    Phi = contrib.evaluator.eval_columns(X)

    d = X - position
    r = torch.linalg.norm(d, dim=1).clamp_min(1e-9)
    r3 = r * r * r
    expected = torch.stack(
        (
            float(K_E) * d[:, 0] / r3,
            float(K_E) * d[:, 1] / r3,
            float(K_E) * d[:, 2] / r3,
        ),
        dim=1,
    )

    assert torch.allclose(Phi, expected, rtol=1e-5, atol=1e-6)
    # Weighted potential with coefficient vector p
    moment = torch.tensor([1.0, -0.5, 0.25], device=device)
    pot = contrib.evaluator.matvec(moment, X)
    manual = (
        expected[:, 0] * moment[0]
        + expected[:, 1] * moment[1]
        + expected[:, 2] * moment[2]
    )
    assert torch.allclose(pot, manual, rtol=1e-5, atol=1e-6)
    assert Phi.device.type == ctx.device.type
