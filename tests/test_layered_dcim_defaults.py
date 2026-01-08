import torch

from electrodrive.flows.schemas import SCHEMA_REAL_POINT
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl import AddBranchCutBlock, AddPoleBlock, Program
from electrodrive.gfn.integration import compile_program_to_basis
from electrodrive.images.basis import DCIMBranchCutImageBasis, DCIMPoleImageBasis
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip


def _layered_spec() -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def _make_payload(device: torch.device) -> ParamPayload:
    u_latent = torch.zeros((1, 2, 4), device=device, dtype=torch.float32)
    node_mask = torch.tensor([[True, True]], device=device)
    schema_ids = torch.tensor([[SCHEMA_REAL_POINT, SCHEMA_REAL_POINT]], device=device, dtype=torch.long)
    return ParamPayload(
        u_latent=u_latent,
        node_mask=node_mask,
        dim_mask=None,
        schema_ids=schema_ids,
        node_to_token=[[0, 1]],
        seed=123,
        config_hash="test_layered_dcim_defaults",
        device=device,
        dtype=torch.float32,
    )


def test_layered_dcim_defaults_enforce_complex_depth() -> None:
    ensure_cuda_available_or_skip("layered dcim defaults")
    device = torch.device("cuda")
    spec = _layered_spec()
    program = Program(
        nodes=(
            AddPoleBlock(interface_id=0, n_poles=2, schema_id=SCHEMA_REAL_POINT),
            AddBranchCutBlock(interface_id=0, approx_type="pade", budget=2, schema_id=SCHEMA_REAL_POINT),
        )
    )
    payload = _make_payload(device)
    elems, _, _ = compile_program_to_basis(program, spec, device, param_payload=payload)

    assert len(elems) == 2
    assert isinstance(elems[0], DCIMPoleImageBasis)
    assert isinstance(elems[1], DCIMBranchCutImageBasis)
    for elem in elems:
        z_imag = elem.params.get("z_imag")
        assert torch.is_tensor(z_imag)
        assert torch.isfinite(z_imag).all()
        assert torch.all(z_imag > 0)
