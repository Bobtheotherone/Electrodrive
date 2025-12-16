import math

import pytest

from electrodrive.layers.dcim_types import DCIMBlock, DCIMCertificate, ComplexImageTerm
from electrodrive.layers.poles import PoleSearchConfig, find_poles
from electrodrive.layers.spectral_kernels import SpectralKernelSpec
from electrodrive.layers.stack import Layer, LayerInterface, LayerStack, layerstack_from_spec
from electrodrive.orchestration.parser import CanonicalSpec


def test_find_poles_disabled_returns_empty():
    cfg = PoleSearchConfig(max_poles=0)
    dummy_stack = LayerStack(
        layers=(Layer("a", eps=1.0 + 0j, z_min=0.0, z_max=1.0),),
        interfaces=tuple(),
        z_bounds=(1.0, 0.0),
    )
    poles = find_poles(dummy_stack, SpectralKernelSpec(0, 0), cfg, device="cuda")
    assert poles == []


def test_find_poles_raises_without_placeholder():
    cfg = PoleSearchConfig(max_poles=2)
    dummy_stack = LayerStack(
        layers=(Layer("a", eps=1.0 + 0j, z_min=0.0, z_max=1.0),),
        interfaces=tuple(),
        z_bounds=(1.0, 0.0),
    )
    with pytest.raises(NotImplementedError):
        find_poles(dummy_stack, SpectralKernelSpec(0, 0), cfg, device="cuda")


def test_dcim_block_serialization_roundtrip():
    spec = CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1, -1, -1], [1, 1, 1]]},
            "conductors": [],
            "dielectrics": [
                {"name": "r1", "epsilon": 1.0, "z_min": 0.0, "z_max": math.inf},
                {"name": "r2", "epsilon": 2.0, "z_min": -1.0, "z_max": 0.0},
            ],
            "charges": [],
            "BCs": "dielectric_interfaces",
        }
    )
    stack = layerstack_from_spec(spec)
    cert = DCIMCertificate(
        k_grid=(0.1, 1.0),
        fit_residual_L2=0.1,
        fit_residual_Linf=0.1,
        spatial_check_rel_L2=0.2,
        spatial_check_rel_Linf=0.2,
        stable=False,
        meta={},
    )
    block = DCIMBlock(
        stack=stack,
        kernel=SpectralKernelSpec(0, 0),
        poles=tuple(),
        images=(ComplexImageTerm(depth=0.5 + 0j, weight=0.1 + 0j),),
        certificate=cert,
    )
    blob = block.to_json()
    block2 = DCIMBlock.from_json(blob)
    assert block2.stack.z_bounds
    assert block2.stack.layer_index(0.1) == 0
    assert block2.stack.layer_index(-0.5) == 1
    assert block2.stack.thickness(1) == pytest.approx(1.0)
