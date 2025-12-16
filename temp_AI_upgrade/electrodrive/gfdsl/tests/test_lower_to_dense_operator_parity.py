import torch

from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    DipoleNode,
    ImageLadderNode,
    IntegerSoftRoundTransform,
    MirrorAcrossPlaneNode,
    Param,
    RealImageChargeNode,
    SoftplusTransform,
    SumNode,
)
from electrodrive.gfdsl.compile import CompileContext


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_dense_vs_operator_parity():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")

    real_node = RealImageChargeNode(
        params={"position": Param(torch.tensor([0.0, 0.0, 0.5], device=device))}
    )
    dipole_node = DipoleNode(
        params={
            "position": Param(torch.tensor([0.1, -0.1, 0.2], device=device)),
        }
    )
    mirror_real = MirrorAcrossPlaneNode(
        children=(real_node,),
        params={"z0": Param(torch.tensor(0.0, device=device))},
        meta={"sign": -1},
    )
    ladder_dipole = ImageLadderNode(
        children=(dipole_node,),
        params={
            "step": Param(torch.tensor(0.05, device=device), transform=SoftplusTransform(min=1e-3)),
            "count": Param(torch.tensor(3.0, device=device), transform=IntegerSoftRoundTransform(min_value=1, max_value=8)),
            "decay": Param(torch.tensor(0.0, device=device)),
        },
        meta={"axis": "z", "direction": 1},
    )
    complex_child = ComplexImageChargeNode(
        params={
            "x": Param(0.05),
            "y": Param(-0.02),
            "a": Param(0.3),
            "b": Param(raw=torch.tensor(0.25), transform=SoftplusTransform(min=1e-3)),
        }
    )
    pair_node = ConjugatePairNode(children=(complex_child,))

    program = SumNode(children=(mirror_real, ladder_dipole, pair_node))
    contrib = program.lower(ctx)

    X = torch.randn(6, 3, device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)
    w = torch.randn(contrib.evaluator.K, device=device, dtype=ctx.dtype)
    y_dense = Phi @ w
    y_operator = contrib.evaluator.matvec(w, X)

    r = torch.randn(X.shape[0], device=device, dtype=ctx.dtype)
    g_dense = Phi.transpose(0, 1) @ r
    g_operator = contrib.evaluator.rmatvec(r, X)

    assert torch.allclose(y_dense, y_operator, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_dense, g_operator, rtol=1e-5, atol=1e-6)
    assert Phi.device.type == ctx.device.type
