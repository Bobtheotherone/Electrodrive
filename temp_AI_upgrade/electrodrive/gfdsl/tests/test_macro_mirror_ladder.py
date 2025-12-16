import torch

from electrodrive.gfdsl.ast import (
    DipoleNode,
    IntegerSoftRoundTransform,
    ImageLadderNode,
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


def test_mirror_across_plane_matches_manual_sum():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    real_node = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.1, -0.1, 0.3], device=device)),
        }
    )
    dipole_node = DipoleNode(
        params={
            "position": Param(torch.tensor([-0.05, 0.2, -0.15], device=device)),
        }
    )
    base_program = SumNode(children=(real_node, dipole_node))
    mirror_node = MirrorAcrossPlaneNode(
        children=(base_program,),
        params={"z0": Param(torch.tensor(0.0, device=device))},
        meta={"sign": -1},
    )

    contrib = mirror_node.lower(ctx)
    X = torch.randn(5, 3, device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)

    # Manual construction: Phi_child + sign * Phi_reflected with z -> -z, pz -> -pz
    z0 = 0.0
    real_pos = real_node.params["position"].value(device=device, dtype=ctx.dtype).reshape(1, 3)
    real_reflected = real_pos.clone()
    real_reflected[..., 2] = 2 * z0 - real_reflected[..., 2]
    real_phi = coulomb_potential_real(X, real_pos)
    real_phi_ref = coulomb_potential_real(X, real_reflected)
    real_total = real_phi + (-1.0) * real_phi_ref

    dip_pos = dipole_node.params["position"].value(device=device, dtype=ctx.dtype).reshape(1, 3)
    dip_ref_pos = dip_pos.clone()
    dip_ref_pos[..., 2] = 2 * z0 - dip_ref_pos[..., 2]
    dip_phi = dipole_basis_real(X, dip_pos)
    dip_phi_ref = dipole_basis_real(X, dip_ref_pos)
    dip_phi_ref = torch.stack(
        (dip_phi_ref[:, 0], dip_phi_ref[:, 1], -dip_phi_ref[:, 2]),
        dim=1,
    )
    dip_total = dip_phi + (-1.0) * dip_phi_ref

    expected = torch.cat((real_total, dip_total), dim=1)
    assert torch.allclose(Phi, expected, rtol=1e-5, atol=1e-6)
    assert Phi.device.type == ctx.device.type


def test_image_ladder_shared_slot_matches_shifted_sum():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32, eval_backend="dense")
    ladder_child = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.05, -0.05, 0.2], device=device)),
        }
    )
    ladder_node = ImageLadderNode(
        children=(ladder_child,),
        params={
            "step": Param(torch.tensor(0.1, device=device), transform=SoftplusTransform(min=1e-3)),
            "count": Param(
                torch.tensor(3.0, device=device),
                transform=IntegerSoftRoundTransform(min_value=1, max_value=8),
            ),
            "decay": Param(torch.tensor(0.0, device=device)),
        },
        meta={"axis": "z", "direction": 1},
    )
    contrib = ladder_node.lower(ctx)

    X = torch.tensor([[0.1, 0.0, 0.05], [-0.1, 0.1, 0.3]], device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)

    # Manual ladder sum for three rungs with unit decay.
    step = ladder_node.params["step"].value(device=device, dtype=ctx.dtype).item()
    count_val = int(
        torch.round(ladder_node.params["count"].value(device=device, dtype=ctx.dtype))
        .clamp(min=1, max=256)
        .item()
    )
    decay = ladder_node.params["decay"].value(device=device, dtype=ctx.dtype).reshape(-1)[0]
    pos0 = ladder_child.params["position"].value(device=device, dtype=ctx.dtype).reshape(1, 3)
    ladder_terms = []
    for k in range(count_val):
        pos_k = pos0 + torch.tensor([[0.0, 0.0, step * k]], device=device, dtype=ctx.dtype)
        weight = torch.exp(-decay * torch.tensor(float(k), device=device, dtype=ctx.dtype))
        ladder_terms.append(coulomb_potential_real(X, pos_k) * weight)
    ladder_total = torch.sum(torch.stack(ladder_terms, dim=0), dim=0)

    assert torch.allclose(Phi[:, :1], ladder_total, rtol=1e-5, atol=1e-6)
    assert Phi.device.type == ctx.device.type


def test_macro_group_policy_override_sets_family_and_motif():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32)
    base = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, 0.1], device=device)),
        }
    )
    mirror = MirrorAcrossPlaneNode(
        children=(base,),
        params={"z0": Param(0.0)},
        meta={"group_policy": "override", "motif_index": 3},
    )
    contrib = mirror.lower(ctx)
    for slot in contrib.slots:
        assert slot.group_info is not None
        assert slot.group_info.family_name == "mirror_plane"
        assert slot.group_info.motif_index == 3
