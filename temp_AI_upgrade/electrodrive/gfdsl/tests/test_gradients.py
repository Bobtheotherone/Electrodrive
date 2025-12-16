import torch

from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    Param,
    SoftplusTransform,
)
from electrodrive.gfdsl.compile import CompileContext


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_complex_pair_gradients_wrt_parameters():
    torch.manual_seed(1234)
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")

    child = ComplexImageChargeNode(
        params={
            "x": Param(0.0),
            "y": Param(0.0),
            "a": Param(0.15),
            "b": Param(raw=torch.tensor(0.2), transform=SoftplusTransform(min=1e-3)),
        }
    )
    node = ConjugatePairNode(children=(child,))
    contrib = node.lower(ctx)

    X = torch.randn(12, 3, device=device, dtype=ctx.dtype)
    w = torch.tensor([1.0, 0.4], device=device, dtype=ctx.dtype)

    V = contrib.evaluator.matvec(w, X)
    loss = (V ** 2).mean()
    loss.backward()

    a_grad = child.params["a"].raw.grad
    b_grad = child.params["b"].raw.grad

    assert a_grad is not None and torch.isfinite(a_grad).all()
    assert b_grad is not None and torch.isfinite(b_grad).all()
    assert a_grad.abs().sum() > 0
    assert b_grad.abs().sum() > 0
    assert V.device.type == ctx.device.type
