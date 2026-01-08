import pytest
import torch

from electrodrive.gfdsl.ast import BranchCutApproxNode, InterfacePoleNode, Param, SumNode
from electrodrive.gfdsl.compile import CompileContext, lower_program


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for layered GFDSL tests"
)


def _laplacian_fd(eval_fn, pts: torch.Tensor, h: float = 2e-2) -> torch.Tensor:
    V0 = eval_fn(pts)
    lap = torch.zeros_like(V0)
    eye = torch.eye(3, device=pts.device, dtype=pts.dtype) * h
    for dim in range(3):
        offset = eye[dim].unsqueeze(0)
        plus = eval_fn(pts + offset)
        minus = eval_fn(pts - offset)
        lap = lap + (plus - 2.0 * V0 + minus) / (h * h)
    return lap


def test_layered_primitives_pde_sanity():
    device = torch.device("cuda")
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")

    pole = InterfacePoleNode(
        params={
            "mode_id": Param(torch.tensor(0.0, device=device)),
            "k_pole": Param(torch.tensor([0.4, 0.8], device=device)),
            "residue": Param(torch.tensor([1e-9, -5e-10], device=device)),
        }
    )
    branch = BranchCutApproxNode(
        params={
            "depths": Param(torch.tensor([0.25, 0.6], device=device)),
            "weights": Param(torch.tensor([2e-10, 1e-10], device=device)),
        },
        meta={"kind": "exp_sum"},
    )

    program = SumNode(children=(pole, branch))
    contrib = lower_program(program, ctx)
    weights = torch.ones(contrib.evaluator.K, device=device, dtype=ctx.dtype)

    def eval_fn(pts: torch.Tensor) -> torch.Tensor:
        return contrib.evaluator.matvec(weights, pts)

    pts = torch.randn(12, 3, device=device, dtype=ctx.dtype) * 0.2
    pts[:, 2] = pts[:, 2] + 1.0

    lap = _laplacian_fd(eval_fn, pts)
    assert torch.isfinite(lap).all()
    assert float(torch.max(torch.abs(lap)).item()) < 5e-2
