import pytest
import torch

from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    Param,
    SoftplusTransform,
)
from electrodrive.gfdsl.compile import CompileContext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for complex-pair harmonicity test"
)


def _laplacian(u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    lap = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
    for i in range(3):
        grad_i = grad[:, i]
        second = torch.autograd.grad(grad_i.sum(), X, create_graph=True)[0][:, i]
        lap = lap + second
    return lap


def test_conjugate_pair_is_harmonic_cuda():
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float64
    ctx = CompileContext(device=device, dtype=dtype, eval_backend="dense")

    child = ComplexImageChargeNode(
        params={
            "x": Param(0.12),
            "y": Param(-0.18),
            "a": Param(0.25),
            "b": Param(
                raw=torch.tensor(0.3, device=device, dtype=dtype),
                transform=SoftplusTransform(min=1e-3),
            ),
        }
    )
    node = ConjugatePairNode(children=(child,))
    contrib = node.lower(ctx)

    X = torch.randn(64, 3, device=device, dtype=dtype) * 0.6 + torch.tensor(
        [0.4, -0.2, 0.5], device=device, dtype=dtype
    )
    X.requires_grad_(True)

    Phi = contrib.evaluator.eval_columns(X)
    u = Phi @ torch.ones(Phi.shape[1], device=device, dtype=dtype)

    lap = _laplacian(u, X)
    assert lap.abs().mean() < 1e-3
    assert lap.abs().max() < 1e-2
