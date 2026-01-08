import torch

from electrodrive.experiments.layered_sampling import (
    parse_layered_interfaces,
    sample_layered_interior,
    sample_layered_interface_pairs,
)
from electrodrive.orchestration.parser import CanonicalSpec


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


def _run_sampling(device: torch.device) -> None:
    dtype = torch.float32
    spec = _layered_spec()
    exclusion_radius = 0.05
    interface_band = 0.02
    domain_scale = 1.0

    interior = sample_layered_interior(
        spec,
        256,
        device=device,
        dtype=dtype,
        seed=123,
        exclusion_radius=exclusion_radius,
        interface_band=interface_band,
        domain_scale=domain_scale,
    )
    assert interior.shape == (256, 3)
    assert interior.dtype == dtype
    assert interior.device.type == device.type
    assert torch.any(interior[:, 2] > 0.0)
    assert torch.any(interior[:, 2] < 0.0)
    assert torch.any((interior[:, 2] > -0.3) & (interior[:, 2] < 0.0))

    charge = torch.tensor([0.0, 0.0, 0.2], device=device, dtype=dtype)
    dist = torch.linalg.norm(interior - charge, dim=1)
    assert torch.all(dist > exclusion_radius * 0.999)

    interfaces = parse_layered_interfaces(spec)
    up, dn = sample_layered_interface_pairs(
        spec,
        n_xy=8,
        device=device,
        dtype=dtype,
        seed=321,
        delta=0.01,
        domain_scale=domain_scale,
    )
    assert up.shape[1] == 3
    assert dn.shape[1] == 3
    for z in interfaces:
        z_up = torch.tensor(z + 0.01, device=device, dtype=dtype)
        z_dn = torch.tensor(z - 0.01, device=device, dtype=dtype)
        assert torch.any(torch.isclose(up[:, 2], z_up, atol=1e-6))
        assert torch.any(torch.isclose(dn[:, 2], z_dn, atol=1e-6))


def test_layered_sampling_alignment_cpu() -> None:
    _run_sampling(torch.device("cpu"))


def test_layered_sampling_alignment_cuda() -> None:
    if not torch.cuda.is_available():
        return
    _run_sampling(torch.device("cuda"))
