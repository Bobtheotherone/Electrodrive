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

    if X.dtype == torch.float64:
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        real_dtype = torch.float32
        complex_dtype = torch.complex64

    dx = dx.to(real_dtype)
    dy = dy.to(real_dtype)
    dz = dz.to(real_dtype)
    b = b.to(real_dtype)

    rho2 = dx * dx + dy * dy
    dzc = dz.to(complex_dtype) - 1j * b.to(complex_dtype)
    rc = torch.sqrt(rho2.to(complex_dtype) + dzc * dzc)

    ke = _ke_tensor(X.device, real_dtype).to(complex_dtype)
    gc = ke / rc

    phi1 = 2.0 * gc.real
    phi2 = -2.0 * gc.imag

    stacked = torch.stack((phi1, phi2), dim=-1)  # (N, K, 2)
    out = stacked.reshape(X.shape[0], -1)
    if out.dtype != X.dtype:
        out = out.to(dtype=X.dtype)
    return out
