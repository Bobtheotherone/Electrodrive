import torch

from electrodrive.gfdsl.ast import (
    BranchCutApproxNode,
    ComplexImageChargeNode,
    ConjugatePairNode,
    DCIMBlockNode,
    InterfacePoleNode,
    Param,
    SoftplusTransform,
)
from electrodrive.gfdsl.compile import CompileContext


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def test_layered_nodes_validate_and_lower():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32)
    pole = InterfacePoleNode(
        params={
            "mode_id": Param(torch.tensor(1.0, device=device)),
            "k_pole": Param(torch.tensor(0.5, device=device)),
            "residue": Param(torch.tensor(0.25, device=device)),
        }
    )
    pole.validate(ctx)
    payload = pole.to_json_dict()
    pole_roundtrip = InterfacePoleNode.from_json_dict(payload)
    pole_roundtrip.validate(ctx)
    contrib = pole_roundtrip.lower(ctx)
    X = torch.randn(4, 3, device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)
    assert Phi.shape == (4, contrib.evaluator.K)
    assert Phi.device.type == ctx.device.type

    branch = BranchCutApproxNode(
        params={
            "depths": Param(torch.tensor([0.2, 0.5], device=device)),
            "weights": Param(torch.tensor([1.0, 0.5], device=device)),
        },
        meta={"kind": "exp_sum"},
    )
    branch.validate(ctx)
    branch_payload = branch.to_json_dict()
    branch_rt = BranchCutApproxNode.from_json_dict(branch_payload)
    branch_rt.validate(ctx)
    contrib_branch = branch_rt.lower(ctx)
    Phi_branch = contrib_branch.evaluator.eval_columns(X)
    assert Phi_branch.shape == (4, contrib_branch.evaluator.K)
    assert Phi_branch.device.type == ctx.device.type


def test_dcim_block_partial_lower():
    device = _device()
    ctx = CompileContext(device=device, dtype=torch.float32)

    complex_child = ComplexImageChargeNode(
        params={
            "x": Param(0.0),
            "y": Param(0.0),
            "a": Param(0.25),
            "b": Param(raw=torch.tensor(0.3), transform=SoftplusTransform(min=1e-3)),
        }
    )
    pair = ConjugatePairNode(children=(complex_child,))
    dcim_supported = DCIMBlockNode(poles=tuple(), images=(pair,), branchcut=None)
    dcim_supported.validate(ctx)
    contrib = dcim_supported.lower(ctx)
    X = torch.randn(4, 3, device=device, dtype=ctx.dtype)
    Phi = contrib.evaluator.eval_columns(X)
    assert Phi.shape[1] == contrib.evaluator.K == 2
    assert Phi.device.type == ctx.device.type

    pole = InterfacePoleNode(
        params={"mode_id": Param(0.0), "k_pole": Param(1.0), "residue": Param(1.0)}
    )
    branch = BranchCutApproxNode(
        meta={"kind": "quadrature_hankel"},
        params={
            "depths": Param(torch.tensor([0.4, 0.8], device=device)),
            "weights": Param(torch.tensor([1.0, 0.2], device=device)),
        },
    )
    dcim_with_branchcut = DCIMBlockNode(
        poles=(pole,),
        images=(pair,),
        branchcut=branch,
    )
    dcim_with_branchcut.validate(ctx)
    contrib_dcim = dcim_with_branchcut.lower(ctx)
    Phi_dcim = contrib_dcim.evaluator.eval_columns(X)
    assert Phi_dcim.shape == (4, contrib_dcim.evaluator.K)
    assert Phi_dcim.device.type == ctx.device.type
