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


def _boom(*args, **kwargs):
    raise AssertionError("eval_columns should not be called in operator mode")


def _patch_eval_columns(evaluator):
    evaluator.eval_columns = _boom  # type: ignore[assignment]
    child = getattr(evaluator, "_child_evaluator", None)
    if child is not None:
        child.eval_columns = _boom  # type: ignore[assignment]


def test_operator_path_avoids_dense_mirror_and_ladder():
    device = _device()
    dense_ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    op_ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")

    real = RealImageChargeNode(params={"position": Param(torch.tensor([0.0, 0.0, 0.5], device=device))})
    dipole = DipoleNode(params={"position": Param(torch.tensor([0.1, -0.1, 0.2], device=device))})
    complex_child = ComplexImageChargeNode(
        params={
            "x": Param(0.05),
            "y": Param(-0.02),
            "a": Param(0.3),
            "b": Param(raw=torch.tensor(0.25), transform=SoftplusTransform(min=1e-3)),
        }
    )
    pair = ConjugatePairNode(children=(complex_child,))
    mirror_program = MirrorAcrossPlaneNode(
        children=(SumNode(children=(real, dipole, pair)),),
        params={"z0": Param(torch.tensor(0.0, device=device))},
        meta={"sign": -1},
    )

    ladder_child_real = RealImageChargeNode(params={"position": Param(torch.tensor([0.0, 0.0, 0.1], device=device))})
    ladder_child_dipole = DipoleNode(params={"position": Param(torch.tensor([0.05, 0.05, -0.1], device=device))})
    ladder_program = ImageLadderNode(
        children=(SumNode(children=(ladder_child_real, ladder_child_dipole)),),
        params={
            "step": Param(torch.tensor(0.05, device=device), transform=SoftplusTransform(min=1e-3)),
            "count": Param(torch.tensor(3.0, device=device), transform=IntegerSoftRoundTransform(min_value=1, max_value=8)),
            "decay": Param(torch.tensor(0.1, device=device)),
        },
        meta={"axis": "z", "direction": 1},
    )

    dense_mirror = mirror_program.lower(dense_ctx)
    dense_ladder = ladder_program.lower(dense_ctx)

    op_mirror = mirror_program.lower(op_ctx)
    op_ladder = ladder_program.lower(op_ctx)

    _patch_eval_columns(op_mirror.evaluator)
    _patch_eval_columns(op_ladder.evaluator)

    X = torch.randn(5, 3, device=device, dtype=torch.float32)
    w_mirror = torch.randn(op_mirror.evaluator.K, device=device, dtype=torch.float32)
    r_mirror = torch.randn(X.shape[0], device=device, dtype=torch.float32)

    y_mirror_op = op_mirror.evaluator.matvec(w_mirror, X)
    g_mirror_op = op_mirror.evaluator.rmatvec(r_mirror, X)

    y_mirror_dense = dense_mirror.evaluator.matvec(w_mirror, X)
    g_mirror_dense = dense_mirror.evaluator.rmatvec(r_mirror, X)

    assert torch.allclose(y_mirror_op, y_mirror_dense, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_mirror_op, g_mirror_dense, rtol=1e-5, atol=1e-6)

    w_ladder = torch.randn(op_ladder.evaluator.K, device=device, dtype=torch.float32)
    r_ladder = torch.randn(X.shape[0], device=device, dtype=torch.float32)

    y_ladder_op = op_ladder.evaluator.matvec(w_ladder, X)
    g_ladder_op = op_ladder.evaluator.rmatvec(r_ladder, X)

    y_ladder_dense = dense_ladder.evaluator.matvec(w_ladder, X)
    g_ladder_dense = dense_ladder.evaluator.rmatvec(r_ladder, X)

    assert torch.allclose(y_ladder_op, y_ladder_dense, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_ladder_op, g_ladder_dense, rtol=1e-5, atol=1e-6)
