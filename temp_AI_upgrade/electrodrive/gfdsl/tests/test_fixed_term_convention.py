import torch

from electrodrive.gfdsl.ast import (
    FixedChargeNode,
    FixedDipoleNode,
    ImageLadderNode,
    IntegerSoftRoundTransform,
    MirrorAcrossPlaneNode,
    Param,
    RealImageChargeNode,
    SoftplusTransform,
    SumNode,
)
from electrodrive.gfdsl.compile import CompileContext
from electrodrive.gfdsl.eval.kernels_real import coulomb_potential_real, dipole_basis_real


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_sum_fixed_and_free_not_double_counted():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")
    fixed_pos = torch.tensor([0.1, 0.0, 0.2], device=device)
    free_pos = torch.tensor([-0.05, 0.02, -0.1], device=device)
    fixed = FixedChargeNode(
        params={"position": Param(fixed_pos), "charge": Param(torch.tensor(2.0, device=device))}
    )
    free = RealImageChargeNode(params={"position": Param(free_pos)})
    program = SumNode(children=(fixed, free))
    contrib = program.lower(ctx)

    X = torch.randn(6, 3, device=device, dtype=ctx.dtype)
    w = torch.tensor([1.5], device=device, dtype=ctx.dtype)
    y_lin = contrib.evaluator.matvec(w, X)
    y_fix = contrib.fixed_term(X) if contrib.fixed_term else torch.zeros_like(y_lin)
    y_total = y_lin + y_fix

    expected = (
        w[0] * coulomb_potential_real(X, free_pos.view(1, 3)).squeeze(-1)
        + 2.0 * coulomb_potential_real(X, fixed_pos.view(1, 3)).squeeze(-1)
    )

    assert torch.allclose(y_total, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y_lin, expected, rtol=1e-5, atol=1e-6)  # ensure fixed not baked into matvec
    assert y_lin.device.type == ctx.device.type


def test_mirror_fixed_term_reflection_present_once():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")
    fixed = FixedChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, 0.2], device=device)),
            "charge": Param(torch.tensor(1.0, device=device)),
        }
    )
    mirror = MirrorAcrossPlaneNode(
        children=(fixed,),
        params={"z0": Param(0.0)},
        meta={"sign": -1},
    )
    contrib = mirror.lower(ctx)

    X = torch.tensor([[0.0, 0.0, 0.05], [0.1, -0.1, 0.3]], device=device, dtype=ctx.dtype)
    y_lin = contrib.evaluator.matvec(torch.zeros(0, device=device, dtype=ctx.dtype), X)
    y_fix = contrib.fixed_term(X) if contrib.fixed_term else torch.zeros(X.shape[0], device=device, dtype=ctx.dtype)
    y_total = y_lin + y_fix

    direct = coulomb_potential_real(X, torch.tensor([[0.0, 0.0, 0.2]], device=device, dtype=ctx.dtype)).squeeze(-1)
    reflected_pos = torch.tensor([[0.0, 0.0, -0.2]], device=device, dtype=ctx.dtype)
    reflected = coulomb_potential_real(X, reflected_pos).squeeze(-1)
    expected = direct + (-1.0) * reflected  # sign = -1

    assert torch.allclose(y_total, expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(y_lin, torch.zeros_like(y_lin))
    assert y_total.device.type == ctx.device.type


def test_mirror_fixed_dipole_term_reflection_present_once():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")
    z0 = 0.0
    pos = torch.tensor([0.0, 0.0, z0 + 0.2], device=device)
    moment = torch.tensor([0.3, -0.1, 0.5], device=device)
    fixed = FixedDipoleNode(
        params={"position": Param(pos), "moment": Param(moment)},
    )
    mirror = MirrorAcrossPlaneNode(
        children=(fixed,),
        params={"z0": Param(z0)},
        meta={"sign": -1},
    )
    contrib = mirror.lower(ctx)

    X = torch.tensor(
        [[0.1, 0.0, 0.05], [0.0, 0.2, 0.3], [-0.15, 0.05, -0.25]],
        device=device,
        dtype=ctx.dtype,
    )
    y_lin = contrib.evaluator.matvec(torch.zeros(0, device=device, dtype=ctx.dtype), X)
    y_fix = contrib.fixed_term(X) if contrib.fixed_term else torch.zeros(X.shape[0], device=device, dtype=ctx.dtype)
    y_total = y_lin + y_fix

    direct = torch.sum(dipole_basis_real(X, pos.view(1, 3)) * moment.view(1, 3), dim=1)
    refl_pos = torch.tensor([0.0, 0.0, 2 * z0 - pos[2]], device=device, dtype=ctx.dtype).view(1, 3)
    refl_moment = moment.clone()
    refl_moment[2] = -refl_moment[2]
    reflected = torch.sum(dipole_basis_real(X, refl_pos) * refl_moment.view(1, 3), dim=1)
    expected = direct + (-1.0) * reflected

    assert torch.allclose(y_lin, torch.zeros_like(y_lin))
    assert torch.allclose(y_total, expected, rtol=1e-5, atol=1e-6)
    assert y_total.device.type == ctx.device.type


def test_ladder_fixed_charge_term_matches_weighted_shifted_sum():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="operator")
    step = 0.1
    decay = 0.2
    count = 3
    fixed = FixedChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, 0.1], device=device)),
            "charge": Param(torch.tensor(2.0, device=device)),
        }
    )
    ladder = ImageLadderNode(
        children=(fixed,),
        params={
            "step": Param(torch.tensor(step, device=device), transform=SoftplusTransform(min=1e-3)),
            "count": Param(torch.tensor(float(count), device=device), transform=IntegerSoftRoundTransform(min_value=1, max_value=8)),
            "decay": Param(torch.tensor(decay, device=device)),
        },
        meta={"axis": "z", "direction": 1},
    )
    contrib = ladder.lower(ctx)

    X = torch.tensor([[0.1, 0.0, 0.05], [-0.1, 0.1, 0.3]], device=device, dtype=ctx.dtype)
    y_lin = contrib.evaluator.matvec(torch.zeros(0, device=device, dtype=ctx.dtype), X)
    y_fix = contrib.fixed_term(X) if contrib.fixed_term else torch.zeros(X.shape[0], device=device, dtype=ctx.dtype)
    y_total = y_lin + y_fix

    step_val = ladder.params["step"].value(device=device, dtype=ctx.dtype).item()
    decay_val = ladder.params["decay"].value(device=device, dtype=ctx.dtype).item()
    count_val = int(
        torch.round(ladder.params["count"].value(device=device, dtype=ctx.dtype))
        .clamp(min=1, max=256)
        .item()
    )
    charge_val = fixed.params["charge"].value(device=device, dtype=ctx.dtype).item()
    pos0 = fixed.params["position"].value(device=device, dtype=ctx.dtype)

    alphas = torch.exp(-decay_val * torch.arange(count_val, device=device, dtype=ctx.dtype))
    expected_terms = []
    for k in range(count_val):
        pos_k = pos0 + torch.tensor([0.0, 0.0, step_val * k], device=device, dtype=ctx.dtype)
        expected_terms.append(alphas[k] * charge_val * coulomb_potential_real(X, pos_k.view(1, 3)).squeeze(-1))
    expected = torch.stack(expected_terms, dim=0).sum(dim=0)

    assert torch.allclose(y_lin, torch.zeros_like(y_lin))
    assert torch.allclose(y_total, expected, rtol=1e-5, atol=1e-6)
    assert y_total.device.type == ctx.device.type
