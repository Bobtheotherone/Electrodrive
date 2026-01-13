import math

import pytest
import torch

from electrodrive.experiments.layered_sampling import parse_layered_interfaces
from electrodrive.experiments.run_discovery import _expand_complex_poles
from electrodrive.images.basis import DCIMPoleImageBasis
from electrodrive.orchestration.parser import CanonicalSpec


def _three_layer_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": math.inf},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.4, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -math.inf, "z_max": -0.4},
            ],
            "charges": [{"type": "point", "q": 1.0, "charge": 1.0, "pos": [0.1, -0.2, 0.2]}],
            "BCs": "dielectric_interfaces",
            "symmetry": ["rot_z"],
            "queries": [],
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_expand_complex_poles_deterministic_cuda() -> None:
    spec = _three_layer_spec()
    device = torch.device("cuda")
    dtype = torch.float32
    elem = DCIMPoleImageBasis(
        {
            "position": torch.tensor([0.1, 0.2, -0.3], device=device, dtype=dtype),
            "z_imag": torch.tensor(0.25, device=device, dtype=dtype),
        }
    )
    planes = parse_layered_interfaces(spec)

    out1 = _expand_complex_poles(
        [elem],
        device=device,
        imag_thresh=1e-6,
        pole_expand_max=3,
        pole_imag_scales=[0.5, 1.0, 2.0],
        pole_depth_steps=[0.0, 0.2, 0.4],
        interface_planes=planes,
        exclusion_radius=0.05,
    )
    out2 = _expand_complex_poles(
        [elem],
        device=device,
        imag_thresh=1e-6,
        pole_expand_max=3,
        pole_imag_scales=[0.5, 1.0, 2.0],
        pole_depth_steps=[0.0, 0.2, 0.4],
        interface_planes=planes,
        exclusion_radius=0.05,
    )

    assert len(out1) == len(out2)
    for a, b in zip(out1, out2):
        pos_a = a.params.get("position")
        pos_b = b.params.get("position")
        assert pos_a is not None and pos_b is not None
        assert torch.allclose(pos_a, pos_b)
        z_a = a.params.get("z_imag")
        z_b = b.params.get("z_imag")
        if z_a is not None and z_b is not None:
            assert torch.allclose(torch.as_tensor(z_a), torch.as_tensor(z_b))
