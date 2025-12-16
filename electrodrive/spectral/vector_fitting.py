from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

_COMPLEX_DTYPES = {torch.complex64, torch.complex128}
if hasattr(torch, "complex32"):
    _COMPLEX_DTYPES.add(torch.complex32)


@dataclass
class VFDiagnostics:
    residuals: List[float] = field(default_factory=list)
    pole_movements: List[float] = field(default_factory=list)
    cond_estimates: List[float] = field(default_factory=list)


def _ensure_cuda_tensor(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dtype not in _COMPLEX_DTYPES and torch.is_complex(x):
        x = torch.real(x)
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.device.type != "cuda":
        raise ValueError("vector_fit requires CUDA tensors (GPU-first).")
    return t


def _poly_from_roots(roots: torch.Tensor) -> torch.Tensor:
    coeff = torch.tensor([1.0 + 0.0j], device=roots.device, dtype=roots.dtype)
    for r in roots:
        new_coeff = torch.zeros(coeff.numel() + 1, device=roots.device, dtype=roots.dtype)
        new_coeff[:-1] = coeff
        new_coeff[1:] += -r * coeff
        coeff = new_coeff
    return coeff


def _poly_from_roots_excluding(roots: torch.Tensor, skip: int) -> torch.Tensor:
    coeff = torch.tensor([1.0 + 0.0j], device=roots.device, dtype=roots.dtype)
    for i, r in enumerate(roots):
        if i == skip:
            continue
        new_coeff = torch.zeros(coeff.numel() + 1, device=roots.device, dtype=roots.dtype)
        new_coeff[:-1] = coeff
        new_coeff[1:] += -r * coeff
        coeff = new_coeff
    return coeff


def _companion_roots(coeff: torch.Tensor) -> torch.Tensor:
    n = coeff.numel() - 1
    if n <= 0:
        return torch.empty(0, device=coeff.device, dtype=coeff.dtype)
    a0 = coeff[0]
    if torch.abs(a0) < 1e-16:
        raise ValueError("Leading polynomial coefficient is zero; cannot build companion matrix.")
    comp = torch.zeros((n, n), device=coeff.device, dtype=coeff.dtype)
    comp[0, :] = -coeff[1:].conj() / a0
    if n > 1:
        comp[1:, :-1] = torch.eye(n - 1, device=coeff.device, dtype=coeff.dtype)
    return torch.linalg.eigvals(comp)


def _enforce_stable(poles: torch.Tensor) -> torch.Tensor:
    re = poles.real
    safe_re = torch.where(re >= 0, -torch.abs(re) - 1e-3, re)
    return safe_re + 1j * poles.imag


def _enforce_conjugate_pairs(poles: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    if poles.numel() == 0:
        return poles
    poles_out = poles.clone()
    used = torch.zeros(poles.numel(), device=poles.device, dtype=torch.bool)
    for i, p in enumerate(poles):
        if used[i]:
            continue
        if torch.abs(p.imag) < tol:
            poles_out[i] = p.real + 0j
            used[i] = True
            continue
        partner = None
        for j in range(poles.numel()):
            if i == j or used[j]:
                continue
            if torch.abs(poles[j] - p.conj()) < 1e-6:
                partner = j
                break
        if partner is None:
            continue
        avg_re = 0.5 * (p.real + poles[partner].real)
        avg_im = 0.5 * (torch.abs(p.imag) + torch.abs(poles[partner].imag))
        new_p = avg_re + 1j * avg_im
        poles_out[i] = new_p
        poles_out[partner] = new_p.conj()
        used[i] = used[partner] = True
    return poles_out


def _enforce_residue_symmetry(poles: torch.Tensor, residues: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    if poles.numel() == 0:
        return residues
    res_out = residues.clone()
    for i, p in enumerate(poles):
        if torch.abs(p.imag) < tol:
            res_out[i] = torch.complex(res_out[i].real, torch.zeros((), device=residues.device, dtype=residues.real.dtype))
            continue
        partner = None
        for j in range(poles.numel()):
            if i == j:
                continue
            if torch.abs(poles[j] - p.conj()) < 1e-6:
                partner = j
                break
        if partner is None:
            continue
        r_avg = 0.5 * (res_out[i] + torch.conj(res_out[partner]))
        res_out[i] = r_avg
        res_out[partner] = torch.conj(r_avg)
    return res_out


def _initial_poles(k: torch.Tensor, M: int) -> torch.Tensor:
    k_max = float(torch.max(torch.real(k)).item())
    k_min = float(torch.min(torch.real(k)).item())
    span = max(k_max - k_min, 1.0)
    re = torch.linspace(-0.2 * span, -span, M, device=k.device, dtype=k.dtype)
    im = torch.zeros_like(re)
    for i in range(M):
        if i % 2 == 1:
            im[i] = 0.5 * span * (0.5 + 0.1 * i)
    poles = re + 1j * im
    return _enforce_conjugate_pairs(_enforce_stable(poles))


def _build_vandermonde(k: torch.Tensor, poles: torch.Tensor) -> torch.Tensor:
    return 1.0 / (k[:, None] - poles[None, :])


def _solve_ls(A: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    sol = torch.linalg.lstsq(A, b)
    residual = torch.linalg.norm(A @ sol.solution - b) / torch.linalg.norm(b).clamp_min(1e-12)
    cond = torch.linalg.cond(A)
    return sol.solution, float(residual.item() if torch.is_tensor(residual) else residual), float(cond.item())


def vector_fit(
    k: torch.Tensor,
    F: torch.Tensor,
    M: Optional[int] = None,
    initial_poles: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    max_iters: int = 8,
    tol: float = 1e-6,
    enforce_conjugates: bool = True,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> Dict[str, object]:
    """
    Clean-room scalar vector fitting on CUDA.
    """
    device = torch.device(device)
    k = _ensure_cuda_tensor(k, device, torch.real(torch.empty((), dtype=dtype)).dtype)
    F = _ensure_cuda_tensor(F, device, dtype)
    if not torch.is_complex(F):
        F = F.to(dtype=dtype)
    N = F.numel()
    if M is None:
        M = max(2, min(8, N // 4))

    poles = initial_poles
    if poles is None:
        poles = _initial_poles(k, M)
    else:
        poles = _ensure_cuda_tensor(poles, device, dtype)
    poles = _enforce_stable(poles)
    if enforce_conjugates:
        poles = _enforce_conjugate_pairs(poles)

    if weights is None:
        weights = torch.ones_like(F.real, device=device, dtype=F.real.dtype)
    else:
        weights = _ensure_cuda_tensor(weights, device, F.real.dtype)

    diagnostics = VFDiagnostics()

    for _ in range(max_iters):
        V = _build_vandermonde(k, poles)
        A = torch.zeros((N, 2 * M + 2), device=device, dtype=dtype)
        A[:, :M] = V
        A[:, M : 2 * M] = -F[:, None] * V
        A[:, 2 * M] = 1.0
        A[:, 2 * M + 1] = k.to(dtype=dtype)

        Aw = A * weights[:, None]
        bw = F * weights

        x, res, cond = _solve_ls(Aw, bw)
        diagnostics.residuals.append(res)
        diagnostics.cond_estimates.append(cond)

        d_coeff = x[M : 2 * M]

        base_poly = _poly_from_roots(poles)
        denom_poly = base_poly.clone()
        for i in range(M):
            term = _poly_from_roots_excluding(poles, i)
            if term.numel() < denom_poly.numel():
                term = torch.cat([torch.zeros(denom_poly.numel() - term.numel(), device=device, dtype=dtype), term], dim=0)
            denom_poly = denom_poly + d_coeff[i] * term
        new_poles = _companion_roots(denom_poly)
        new_poles = _enforce_stable(new_poles)
        if enforce_conjugates:
            new_poles = _enforce_conjugate_pairs(new_poles)

        move = torch.max(torch.abs(new_poles - poles)).item() if new_poles.numel() > 0 else 0.0
        diagnostics.pole_movements.append(move)
        poles = new_poles
        if move < tol:
            break

    V = _build_vandermonde(k, poles)
    A_final = torch.zeros((N, M + 2), device=device, dtype=dtype)
    A_final[:, :M] = V
    A_final[:, M] = 1.0
    A_final[:, M + 1] = k.to(dtype=dtype)
    Aw = A_final * weights[:, None]
    bw = F * weights
    sol, res_final, cond_final = _solve_ls(Aw, bw)
    diagnostics.residuals.append(res_final)
    diagnostics.cond_estimates.append(cond_final)

    residues = sol[:M]
    d = sol[M]
    h = sol[M + 1]

    if enforce_conjugates:
        residues = _enforce_residue_symmetry(poles, residues)

    return {
        "poles": poles,
        "residues": residues,
        "d": d,
        "h": h,
        "diagnostics": diagnostics,
    }
