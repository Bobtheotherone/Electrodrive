import math

import pytest
import torch

from electrodrive.layers import effective_reflection, layerstack_from_spec
from electrodrive.orchestration.parser import CanonicalSpec


def _three_layer_spec(eps2: float = 4.0, h: float = 0.3) -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": math.inf},
                {"name": "slab", "epsilon": eps2, "z_min": -float(h), "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -math.inf, "z_max": -float(h)},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
            "symmetry": ["rot_z"],
            "queries": [],
        }
    )


def _three_layer_reflection(k: torch.Tensor, eps1: complex, eps2: complex, eps3: complex, h: float) -> torch.Tensor:
    R12 = (eps1 - eps2) / (eps1 + eps2)
    R21 = -R12
    R23 = (eps2 - eps3) / (eps2 + eps3)
    T12 = 2.0 * eps2 / (eps1 + eps2)
    T21 = 2.0 * eps1 / (eps1 + eps2)
    exp_h = torch.exp(-2.0 * k * h)
    return R12 + (T12 * T21 * R23 * exp_h) / (1.0 - R21 * R23 * exp_h)


def test_layerstack_validation_and_lookup():
    stack = layerstack_from_spec(_three_layer_spec())
    assert stack.layer_index(0.25) == 0
    assert stack.layer_index(-0.05) == 1
    assert stack.layer_index(-2.0) == 2
    assert math.isinf(stack.thickness(0))
    assert math.isinf(stack.thickness(2))
    assert stack.thickness(1) == pytest.approx(0.3)

    bad_spec = CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]]},
            "conductors": [],
            "dielectrics": [
                {"name": "top", "epsilon": 1.0, "z_min": 0.1, "z_max": 2.0},
                {"name": "mid", "epsilon": 2.0, "z_min": -0.5, "z_max": 0.0},  # gap between 0.1 and 0.0
                {"name": "bottom", "epsilon": 1.0, "z_min": -2.0, "z_max": -0.5},
            ],
            "charges": [],
            "BCs": "dielectric_interfaces",
        }
    )
    with pytest.raises(ValueError):
        layerstack_from_spec(bad_spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU-first recursion test.")
def test_recursion_matches_three_layer_reference():
    dtype = torch.complex128
    device = torch.device("cuda")
    spec = _three_layer_spec(eps2=5.0, h=0.4)
    stack = layerstack_from_spec(spec)

    k = torch.linspace(1e-3, 6.0, 64, device=device, dtype=dtype)
    R_eff = effective_reflection(stack, k, source_region=0, direction="down", device=device, dtype=dtype)
    assert R_eff.device.type == "cuda"

    eps1 = complex(spec.dielectrics[0]["epsilon"])
    eps2 = complex(spec.dielectrics[1]["epsilon"])
    eps3 = complex(spec.dielectrics[2]["epsilon"])
    h = stack.thickness(1)

    R_ref = _three_layer_reflection(k, eps1, eps2, eps3, h)
    torch.testing.assert_close(R_eff, R_ref, rtol=1e-6, atol=1e-8)
