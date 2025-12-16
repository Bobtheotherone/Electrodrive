"""Real-valued evaluation kernels for GFDSL primitives."""

from __future__ import annotations

import torch

from electrodrive.utils.config import K_E


def _ke_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(float(K_E), device=device, dtype=dtype)


def _clamp_r(r: torch.Tensor) -> torch.Tensor:
    # Length scale placeholder (1.0) per AGENTS instructions.
    return r.clamp_min(1e-9)


def coulomb_potential_real(X: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Compute Coulomb potential columns for real image charges."""
    d = X[:, None, :] - pos[None, :, :]
    r = torch.linalg.norm(d, dim=-1)
    r = _clamp_r(r)
    ke = _ke_tensor(X.device, X.dtype)
    return ke / r


def dipole_basis_real(X: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Compute dipole basis columns (px, py, pz) for each position."""
    d = X[:, None, :] - pos[None, :, :]
    r = torch.linalg.norm(d, dim=-1)
    r = _clamp_r(r)
    r3 = r * r * r
    ke = _ke_tensor(X.device, X.dtype)

    dx = d[..., 0]
    dy = d[..., 1]
    dz = d[..., 2]

    phi_x = ke * dx / r3
    phi_y = ke * dy / r3
    phi_z = ke * dz / r3

    stacked = torch.stack((phi_x, phi_y, phi_z), dim=-1)  # (N, K, 3)
    return stacked.reshape(X.shape[0], -1)
