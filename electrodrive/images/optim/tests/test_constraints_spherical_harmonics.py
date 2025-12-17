import math
import torch
import pytest

from electrodrive.images.optim.bases.spherical_harmonics import SphericalHarmonicsConstraintOp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for spherical harmonics constraints")
def test_spherical_harmonics_constraint_modes():
    device = torch.device("cuda")
    dtype = torch.float32
    n_theta, n_phi = 12, 24

    theta = torch.linspace(0.0, math.pi, n_theta, device=device, dtype=dtype)
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi, device=device, dtype=dtype)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.sin(theta_grid) * torch.cos(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(theta_grid)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    op = SphericalHarmonicsConstraintOp(points=points, lmax=3, device=device, dtype=dtype)
    idx = op.mode_map[(2, 1)]

    coeffs = torch.zeros(op._basis.shape[1], device=device, dtype=op._basis.dtype)
    coeffs[idx] = 1.0 + 0.0j
    r = op.adjoint(coeffs)
    out = op.apply(r)

    n_modes = op._basis.shape[1]
    out_complex = out[:n_modes] + 1j * out[n_modes:]
    mags = torch.abs(out_complex)

    assert torch.isfinite(r).all()
    assert float(mags[idx].item()) >= 0.5 * float(mags.max().item())
