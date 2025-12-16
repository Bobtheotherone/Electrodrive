"""Real-arithmetic kernels for complex image charges."""

from __future__ import annotations

import torch

from electrodrive.utils.config import K_E


def _ke_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(float(K_E), device=device, dtype=dtype)


def complex_conjugate_pair_columns(X: torch.Tensor, xyab: torch.Tensor) -> torch.Tensor:
    """Return real-valued columns (phi1, phi2) for complex conjugate image charges.

    Args:
        X: [N, 3] query points.
        xyab: [K, 4] parameters (x, y, a, b) per complex charge with b>0.
    """
    dx = X[:, None, 0] - xyab[None, :, 0]
    dy = X[:, None, 1] - xyab[None, :, 1]
    dz = X[:, None, 2] - xyab[None, :, 2]
    b = xyab[None, :, 3]

    rho2 = dx * dx + dy * dy
    u = rho2 + dz * dz - b * b
    v = -2.0 * b * dz

    r = torch.sqrt(u * u + v * v)
    sr = torch.sqrt(torch.clamp((r + u) * 0.5, min=0.0))
    si = torch.sign(v) * torch.sqrt(torch.clamp((r - u) * 0.5, min=0.0))

    denom = (sr * sr + si * si).clamp_min(1e-12)
    inv_real = sr / denom
    inv_imag = -si / denom

    ke = _ke_tensor(X.device, X.dtype)
    phi1 = 2.0 * ke * inv_real
    phi2 = -2.0 * ke * inv_imag

    stacked = torch.stack((phi1, phi2), dim=-1)  # (N, K, 2)
    return stacked.reshape(X.shape[0], -1)
