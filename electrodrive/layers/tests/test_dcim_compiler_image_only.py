import math

import pytest
import torch

from electrodrive.layers import DCIMCompilerConfig, SpectralKernelSpec, compile_dcim, layerstack_from_spec
from electrodrive.layers.dcim_compiler import _image_domain_potential, _reflected_potential
from electrodrive.orchestration.parser import CanonicalSpec


def _three_layer_spec(eps2: float = 5.0, h: float = 0.4) -> CanonicalSpec:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_compile_dcim_image_only_stable_and_image_valid():
    device = torch.device("cuda")
    dtype = torch.complex128
    spec = _three_layer_spec(eps2=4.0, h=0.4)
    stack = layerstack_from_spec(spec)
    kernel = SpectralKernelSpec(source_region=0, obs_region=0, component="potential", bc_kind="dielectric_interfaces")
    cfg = DCIMCompilerConfig(
        k_min=5e-2,
        k_mid=2.0,
        k_max=6.0,
        n_low=64,
        n_mid=64,
        n_high=0,
        log_low=False,
        log_high=False,
        vf_enabled=False,
        vf_for_images=False,
        exp_fit_enabled=True,
        exp_fit_requires_uniform_grid=True,
        exp_N=6,
        spectral_tol=1e-1,
        spatial_tol=1e-1,
        sample_points=[(0.3, 0.6), (0.5, 1.0), (0.2, 0.6)],
        cache_enabled=False,
        device=device,
        dtype=dtype,
        runtime_eval_mode="image_only",
    )

    block = compile_dcim(stack, kernel, cfg)
    assert block.images, "Expected images in image_only mode."
    assert all(p.kind != "vf" for p in block.poles), "No VF poles expected in image_only mode."
    assert block.certificate.stable, "Image-only DCIM block should be stable for this config."

    # Validate image-domain reflected field.
    k = torch.tensor(block.certificate.k_grid, device=device, dtype=dtype)
    eps1 = 1.0
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    rho = torch.tensor([0.3, 0.5, 0.2], device=device, dtype=real_dtype)
    z = torch.tensor([0.6, 1.0, 0.6], device=device, dtype=real_dtype)

    V_pred = _image_domain_potential(block.images, rho, z, z0=0.2, eps1=eps1, q=1.0, device=device, dtype=dtype)
    from electrodrive.layers.rt_recursion import effective_reflection as _eff

    R_ref = _eff(stack, k, source_region=0, direction="down", device=device, dtype=dtype)
    V_reflected = _reflected_potential(k, R_ref, eps1, rho, z, z0=0.2, q=1.0)
    rel_err = torch.linalg.norm(V_pred - V_reflected) / torch.linalg.norm(V_reflected).clamp_min(1e-12)
    assert rel_err < cfg.spatial_tol
