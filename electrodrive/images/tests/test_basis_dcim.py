import math

import pytest
import torch

from electrodrive.images.basis import generate_candidate_basis
from electrodrive.images.basis_dcim import DCIMBlockBasis, dcim_basis_from_block
from electrodrive.layers import DCIMCompilerConfig, SpectralKernelSpec, compile_dcim, layerstack_from_spec
from electrodrive.layers.dcim_compiler import _image_domain_potential, _reflected_potential
from electrodrive.layers.rt_recursion import effective_reflection
from electrodrive.orchestration.parser import CanonicalSpec


def _three_layer_spec(eps2: float = 4.0, h: float = 0.4) -> CanonicalSpec:
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


def _compile_block(spec: CanonicalSpec) -> tuple:
    device = torch.device("cuda")
    dtype = torch.complex128
    stack = layerstack_from_spec(spec)
    kernel = SpectralKernelSpec(source_region=0, obs_region=0, component="potential", bc_kind="dielectric_interfaces")
    cfg = DCIMCompilerConfig(
        k_min=0.05,
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
        spectral_tol=0.3,
        spatial_tol=0.2,
        sample_points=[(0.3, 0.6), (0.5, 1.0), (0.2, 0.6)],
        cache_enabled=False,
        device=device,
        dtype=dtype,
        runtime_eval_mode="image_only",
        source_z=0.2,
        source_charge=1.0,
    )
    block = compile_dcim(stack, kernel, cfg)
    return block, device, dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_dcim_basis_real_matches_image_domain():
    spec = _three_layer_spec(eps2=4.0, h=0.4)
    block, device, dtype = _compile_block(spec)
    assert block.certificate.stable

    elems = dcim_basis_from_block(block)
    real_elems = [e for e in elems if e.component == "real"]
    targets = torch.tensor(
        [[0.3, 0.0, 0.6], [0.5, 0.0, 1.0], [0.2, 0.0, 0.6]],
        device=device,
        dtype=torch.float64,
    )
    V_basis = torch.zeros(targets.shape[0], device=device, dtype=torch.float64)
    for elem in real_elems:
        V_basis = V_basis + elem.potential(targets)

    eps1 = 1.0
    z0 = 0.2
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    rho = torch.linalg.norm(targets[:, :2], dim=1).to(dtype=real_dtype)
    z = targets[:, 2].to(dtype=real_dtype)
    V_ref = _image_domain_potential(block.images, rho, z, z0, eps1, 1.0, device, dtype)

    rel_err = torch.linalg.norm(V_basis - V_ref) / torch.linalg.norm(V_ref).clamp_min(1e-12)
    assert rel_err.item() < 0.2
    assert V_basis.is_cuda
    assert V_basis.dtype in (torch.float32, torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_generate_candidate_basis_includes_dcim_basis():
    spec = _three_layer_spec(eps2=4.0, h=0.4)
    candidates = generate_candidate_basis(spec, ["three_layer_dcim"], n_candidates=64, device="cuda", dtype=torch.float32)
    dcim_elems = [c for c in candidates if isinstance(c, DCIMBlockBasis)]
    assert dcim_elems, "DCIM basis should be generated when requested."
    assert dcim_elems[0].block.certificate.stable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_dcim_basis_against_oracle_speed_and_accuracy():
    spec = _three_layer_spec(eps2=4.0, h=0.4)
    block, device, dtype = _compile_block(spec)
    rho_vals = torch.linspace(0.05, 0.5, 256, device=device)
    z_vals = torch.linspace(0.3, 1.0, 256, device=device)
    targets = torch.zeros((rho_vals.numel(), 3), device=device, dtype=torch.float64)
    targets[:, 0] = rho_vals
    targets[:, 2] = z_vals

    agg_elem = DCIMBlockBasis(block=block, images=block.images, component="real")
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    V_dcim = agg_elem.potential(targets)
    t1.record()
    torch.cuda.synchronize()
    dcim_ms = t0.elapsed_time(t1)

    # Spectral oracle reflected potential on the compiler k-grid.
    k = torch.tensor(block.certificate.k_grid, device=device, dtype=dtype)
    R_ref = effective_reflection(block.stack, k, source_region=0, direction="down", device=device, dtype=dtype)
    rho = torch.linalg.norm(targets[:, :2], dim=1).to(dtype=torch.float64)
    z = targets[:, 2].to(dtype=torch.float64)
    t2 = torch.cuda.Event(enable_timing=True)
    t3 = torch.cuda.Event(enable_timing=True)
    t2.record()
    V_ref_oracle = _reflected_potential(
        k,
        R_ref,
        float(block.stack.layers[0].eps.real),
        rho,
        z,
        z0=0.2,
        q=1.0,
    )
    t3.record()
    torch.cuda.synchronize()
    oracle_ms = t2.elapsed_time(t3)

    rel_err = torch.linalg.norm(V_dcim - V_ref_oracle) / torch.linalg.norm(V_ref_oracle).clamp_min(1e-12)
    assert rel_err.item() < 0.2
    assert dcim_ms < oracle_ms * 2.0
