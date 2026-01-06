import torch

from electrodrive.gfdsl.ast import (
    ComplexImageChargeNode,
    ConjugatePairNode,
    Param,
    SoftplusTransform,
)
from electrodrive.gfdsl.compile import CompileContext
from electrodrive.utils.config import K_E


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_conjugate_pair_matches_complex_reference():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")

    x0 = torch.tensor(0.1, device=device)
    y0 = torch.tensor(-0.05, device=device)
    a = torch.tensor(0.2, device=device)
    b_raw = torch.tensor(0.4, device=device)
    child = ComplexImageChargeNode(
        params={
            "x": Param(x0),
            "y": Param(y0),
            "a": Param(a),
            "b": Param(raw=b_raw, transform=SoftplusTransform(min=1e-3)),
        }
    )
    node = ConjugatePairNode(children=(child,))

    X = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.6], [-0.15, 0.3, 0.25]],
        device=device,
        dtype=ctx.dtype,
    )

    contrib = node.lower(ctx)
    Phi = contrib.evaluator.eval_columns(X)
    assert Phi.shape[1] == 2
    assert torch.isfinite(Phi).all()
    assert Phi.device.type == ctx.device.type

    b = child.params["b"].value(device=device, dtype=ctx.dtype)
    ke = torch.as_tensor(float(K_E), device=device, dtype=torch.cdouble)
    Xc = X.to(dtype=torch.cdouble)
    rho2 = (Xc[:, 0] - x0.to(dtype=torch.cdouble)) ** 2 + (Xc[:, 1] - y0.to(dtype=torch.cdouble)) ** 2
    dz = Xc[:, 2] - a.to(dtype=torch.cdouble)
    r_complex = torch.sqrt(rho2 + (dz - 1j * b.to(dtype=torch.cdouble)) ** 2)
    ref_real = 2.0 * torch.real(ke / r_complex)
    ref_im = -2.0 * torch.imag(ke / r_complex)

    assert torch.allclose(Phi[:, 0], ref_real.real.to(ctx.dtype), rtol=1e-5, atol=1e-5)
    assert torch.allclose(Phi[:, 1], ref_im.real.to(ctx.dtype), rtol=1e-5, atol=1e-5)
