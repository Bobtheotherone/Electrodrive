from __future__ import annotations

from typing import Iterable, Optional, Sequence
import math

import torch


_SH_CACHE: dict[tuple[int, int, torch.dtype, torch.device], dict[str, torch.Tensor | list[tuple[int, int]]]] = {}


def _associated_legendre(lmax: int, x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    p = torch.zeros((lmax + 1, lmax + 1, n), device=x.device, dtype=x.dtype)
    p[0, 0] = 1.0
    if lmax == 0:
        return p
    sqrt1mx2 = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
    for m in range(1, lmax + 1):
        p[m, m] = -(2 * m - 1) * sqrt1mx2 * p[m - 1, m - 1]
    for m in range(0, lmax):
        p[m + 1, m] = x * (2 * m + 1) * p[m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            p[l, m] = ((2 * l - 1) * x * p[l - 1, m] - (l + m - 1) * p[l - 2, m]) / (l - m)
    return p


def _build_sh_basis(
    theta: torch.Tensor,
    phi: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    theta = theta.view(-1)
    phi = phi.view(-1)
    x = torch.cos(theta)
    p = _associated_legendre(lmax, x)

    modes: list[tuple[int, int]] = []
    basis_cols: list[torch.Tensor] = []

    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            m_abs = abs(m)
            norm = math.sqrt((2 * l + 1) / (4.0 * math.pi))
            norm *= math.exp(0.5 * (math.lgamma(l - m_abs + 1) - math.lgamma(l + m_abs + 1)))
            base = p[l, m_abs] * norm
            if m_abs == 0:
                y_lm = base.to(torch.complex64 if base.dtype != torch.float64 else torch.complex128)
            else:
                phase = torch.exp(1j * m_abs * phi)
                y_lm = base * phase
            if m < 0:
                y_lm = ((-1) ** m_abs) * torch.conj(y_lm)
            modes.append((l, m))
            basis_cols.append(y_lm)

    basis = torch.stack(basis_cols, dim=1)
    return basis, modes


def _get_cached_basis(
    theta: torch.Tensor,
    phi: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    key = (int(lmax), int(theta.numel()), theta.dtype, theta.device)
    cached = _SH_CACHE.get(key)
    if cached is not None:
        theta_cached = cached.get("theta")
        phi_cached = cached.get("phi")
        if isinstance(theta_cached, torch.Tensor) and isinstance(phi_cached, torch.Tensor):
            if theta_cached.shape == theta.shape and phi_cached.shape == phi.shape:
                if torch.allclose(theta_cached, theta) and torch.allclose(phi_cached, phi):
                    return cached["basis"], cached["modes"]  # type: ignore[return-value]
    basis, modes = _build_sh_basis(theta, phi, lmax)
    _SH_CACHE[key] = {
        "basis": basis,
        "modes": modes,
        "theta": theta.detach().clone(),
        "phi": phi.detach().clone(),
    }
    return basis, modes


class SphericalHarmonicsConstraintOp:
    def __init__(
        self,
        *,
        points: Optional[torch.Tensor] = None,
        theta: Optional[Sequence[float] | torch.Tensor] = None,
        phi: Optional[Sequence[float] | torch.Tensor] = None,
        lmax: int = 4,
        mode_indices: Optional[Iterable[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_complex: bool = False,
    ) -> None:
        self.device = device or (points.device if points is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (points.dtype if points is not None else torch.float32)
        self.use_complex = bool(use_complex)
        self.lmax = int(lmax)

        if points is None and (theta is None or phi is None):
            raise ValueError("Spherical harmonics constraint requires points or theta/phi.")

        if points is not None:
            pts = points.to(device=self.device, dtype=self.dtype)
            r = torch.linalg.norm(pts, dim=-1).clamp_min(1e-12)
            z = pts[..., 2] / r
            theta_t = torch.acos(torch.clamp(z, -1.0, 1.0))
            phi_t = torch.atan2(pts[..., 1], pts[..., 0])
        else:
            theta_t = torch.as_tensor(theta, device=self.device, dtype=self.dtype).view(-1)
            phi_t = torch.as_tensor(phi, device=self.device, dtype=self.dtype).view(-1)

        basis, modes = _get_cached_basis(theta_t, phi_t, self.lmax)
        self.modes = modes
        self.mode_map = {mode: i for i, mode in enumerate(modes)}

        if mode_indices is None:
            self._basis = basis
        else:
            idx = [self.mode_map[(int(l), int(m))] for l, m in mode_indices]
            self._basis = basis[:, idx]
            self.modes = [modes[i] for i in idx]
            self.mode_map = {mode: i for i, mode in enumerate(self.modes)}

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        r = r.to(device=self.device, dtype=self.dtype).view(-1)
        coeffs = torch.matmul(torch.conj(self._basis).transpose(0, 1), r.to(self._basis.dtype))
        if self.use_complex:
            return coeffs
        return torch.cat([coeffs.real, coeffs.imag], dim=0)

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(c):
            coeffs = c
        else:
            c = c.to(device=self.device, dtype=self.dtype).view(-1)
            n_modes = self._basis.shape[1]
            if c.numel() != 2 * n_modes:
                raise ValueError("Coefficient vector has unexpected length.")
            coeffs = c[:n_modes] + 1j * c[n_modes:]
        r = torch.matmul(self._basis, coeffs.to(self._basis.dtype))
        return r.real
