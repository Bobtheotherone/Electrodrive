import math

import pytest
import torch

from electrodrive.layers import (
    DCIMCompilerConfig,
    SpectralKernelSpec,
    compile_dcim,
    layerstack_from_spec,
)
from electrodrive.layers.dcim_compiler import _image_domain_potential, _reflected_potential
from electrodrive.layers.rt_recursion import effective_reflection
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
def test_compile_dcim_produces_images_and_certificate():
    device = torch.device("cuda")
    dtype = torch.complex128
    spec = _three_layer_spec()
    src_pos = spec.charges[0]["pos"]
    stack = layerstack_from_spec(spec)
    kernel = SpectralKernelSpec(source_region=0, obs_region=0, component="potential", bc_kind="dielectric_interfaces")
    cfg = DCIMCompilerConfig(
        k_min=1e-3,
        k_mid=2.0,
        k_max=6.0,
        n_low=16,
        n_mid=48,
        n_high=16,
        vf_M=4,
        exp_N=3,
        spectral_tol=5e-2,
        spatial_tol=0.25,
        sample_points=[(0.15, 0.3), (0.25, 0.5), (0.35, 0.7)],
        cache_enabled=False,
        device=device,
        dtype=dtype,
        runtime_eval_mode="composite",
        source_pos=(float(src_pos[0]), float(src_pos[1]), float(src_pos[2])),
        source_charge=float(spec.charges[0].get("charge", spec.charges[0].get("q", 1.0))),
    )

    block = compile_dcim(stack, kernel, cfg)
    assert block.images, "Expected at least one complex image term."
    assert block.certificate is not None
    assert block.certificate.fit_residual_L2 >= 0
    assert block.certificate.spatial_check_rel_L2 >= 0
    assert block.certificate.stable, "DCIM block should be stable for moderate contrast config."

    # Reconstruct fitted spectrum from stored poles/images for sanity.
    k = torch.tensor(block.certificate.k_grid, device=device, dtype=dtype)
    eps1 = 1.0
    vf_poles = []
    vf_residues = []
    for p in block.poles:
        if p.kind == "vf":
            vf_poles.append(complex(p.pole))
            vf_residues.append(complex(p.residue))
    if vf_poles:
        vf_poles_t = torch.as_tensor(vf_poles, device=device, dtype=dtype)
        vf_residues_t = torch.as_tensor(vf_residues, device=device, dtype=dtype)
    else:
        vf_poles_t = torch.zeros(0, device=device, dtype=dtype)
        vf_residues_t = torch.zeros(0, device=device, dtype=dtype)
    d = complex(block.certificate.meta.get("vf", {}).get("d", 0.0))
    h = complex(block.certificate.meta.get("vf", {}).get("h", 0.0))
    d_t = torch.as_tensor(d, device=device, dtype=dtype)
    h_t = torch.as_tensor(h, device=device, dtype=dtype)
    exp_A = torch.as_tensor([complex(img.weight) for img in block.images], device=device, dtype=dtype)
    exp_B = torch.as_tensor([complex(img.depth) for img in block.images], device=device, dtype=dtype)

    if vf_poles_t.numel() > 0:
        V = 1.0 / (k[:, None] - vf_poles_t[None, :])
        F_fit = torch.sum(vf_residues_t[None, :] * V, dim=1) + d_t + h_t * k
    else:
        F_fit = d_t + h_t * k
    if exp_A.numel() > 0:
        F_fit = F_fit + torch.sum(exp_A[None, :] * torch.exp(-exp_B[None, :] * k[:, None]), dim=1)

    F_ref = effective_reflection(stack, k, source_region=0, direction="down", device=device, dtype=dtype)
    rel_err = torch.linalg.norm(F_fit - F_ref) / torch.linalg.norm(F_ref).clamp_min(1e-12)
    assert rel_err < 0.4

    # Spatial validation against reflected remainder (image-domain).
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    rho = torch.tensor([0.15, 0.25, 0.35], device=device, dtype=real_dtype)
    z = torch.tensor([0.3, 0.5, 0.7], device=device, dtype=real_dtype)
    z0 = float(block.certificate.meta["source_pos"][2])
    q_src = float(block.certificate.meta.get("source_charge", 1.0))
    z_ref = block.certificate.meta.get("z_ref", None)
    V_fit_img = _image_domain_potential(block.images, rho, z, z0=z0, eps1=eps1, q=q_src, device=device, dtype=dtype, z_ref=z_ref)
    V_vf_ref = torch.zeros_like(rho, device=device, dtype=dtype)
    if vf_poles_t.numel() > 0:
        Vmat = 1.0 / (k[:, None] - vf_poles_t[None, :])
        F_vf = torch.sum(vf_residues_t[None, :] * Vmat, dim=1) + d_t + h_t * k
        V_vf_ref = _reflected_potential(k, F_vf, eps1, rho, z, z0=z0, q=q_src, z_ref=z_ref)
    V_pred = V_vf_ref + V_fit_img
    V_reflected = _reflected_potential(k, F_ref, eps1, rho, z, z0=z0, q=q_src, z_ref=z_ref)
    rel_spatial = torch.linalg.norm(V_pred - V_reflected) / torch.linalg.norm(V_reflected).clamp_min(1e-12)
    assert rel_spatial < cfg.spatial_tol
