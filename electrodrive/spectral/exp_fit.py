from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

_COMPLEX_DTYPES = {torch.complex64, torch.complex128}
if hasattr(torch, "complex32"):
    _COMPLEX_DTYPES.add(torch.complex32)


@dataclass
class ExpFitDiagnostics:
    residual: float
    model_order: int
    eigvals: torch.Tensor = field(default_factory=lambda: torch.empty(0))


def _ensure_cuda_tensor(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dtype not in _COMPLEX_DTYPES and torch.is_complex(x):
        x = torch.real(x)
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.device.type != "cuda":
        raise ValueError("exp_fit requires CUDA tensors (GPU-first).")
    return t


def exp_fit(
    k: torch.Tensor,
    R: torch.Tensor,
    n_terms: int,
    weights: Optional[torch.Tensor] = None,
    include_bias: bool = True,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> Dict[str, object]:
    """
    Estimate exponential sum R(k) ≈ Σ A_n exp(-B_n k) using ESPRIT / matrix pencil.
    """
    device = torch.device(device)
    k = _ensure_cuda_tensor(k, device, torch.real(torch.empty((), dtype=dtype)).dtype)
    R = _ensure_cuda_tensor(R, device, dtype)
    if not torch.is_complex(R):
        R = R.to(dtype=dtype)

    if k.numel() < 2:
        raise ValueError("exp_fit requires at least two k samples.")
    if torch.linalg.norm(R).item() < 1e-12:
        A = torch.zeros(n_terms, device=device, dtype=dtype)
        B = torch.ones(n_terms, device=device, dtype=dtype)
        bias = torch.tensor(0.0, device=device, dtype=dtype)
        diag = ExpFitDiagnostics(residual=0.0, model_order=n_terms, eigvals=torch.zeros(n_terms, device=device, dtype=dtype))
        return {"A": A, "B": B, "bias": bias, "diagnostics": diag}
    delta = torch.diff(k)
    if torch.max(torch.abs(delta - delta[0])) > 1e-6 * torch.abs(delta[0]):
        raise ValueError("exp_fit currently assumes uniformly spaced k-grid.")
    step = delta[0].real

    bias = torch.tensor(0.0, device=device, dtype=dtype)
    R_work = R
    append_bias = include_bias

    N = R_work.numel()
    r_guess = n_terms
    L = min(max(r_guess + 1, N // 2), N - r_guess - 1)
    m = N - L
    # Build Hankel matrices H0, H1 with shift by one sample.
    H0 = torch.zeros((L, m), device=device, dtype=dtype)
    H1 = torch.zeros((L, m), device=device, dtype=dtype)
    for i in range(L):
        H0[i, :] = R_work[i : i + m]
        H1[i, :] = R_work[i + 1 : i + m + 1]

    U, S, Vh = torch.linalg.svd(H0, full_matrices=False)
    sv_thresh = 1e-6 * S[0].real
    r_eff = int(torch.sum(S.real > sv_thresh).item())
    r = max(1, min(n_terms, r_eff, L, m))
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh.conj().transpose(-2, -1)[:, :r]
    Sigma_inv = torch.diag(1.0 / S_r.clamp_min(1e-12)).to(dtype=dtype)
    Phi = U_r.conj().transpose(-2, -1) @ H1 @ V_r @ Sigma_inv
    eigvals_all = torch.linalg.eigvals(Phi)
    idx = torch.argsort(torch.abs(eigvals_all), descending=True)
    eigvals = eigvals_all[idx[:r]]
    B = -torch.log(eigvals) / step
    # Enforce Re(B) > 0 for stability and clamp extreme values.
    B = torch.clamp(B.real, min=1e-9, max=10.0) + 1j * B.imag

    # Solve amplitudes via Vandermonde least squares.
    time = (k - k[0])
    V_base = torch.exp(-B[None, :] * time[:, None])
    Vmat = V_base
    if append_bias:
        Vmat = torch.cat([Vmat, torch.ones((Vmat.shape[0], 1), device=device, dtype=dtype)], dim=1)
    if weights is not None:
        weights = _ensure_cuda_tensor(weights, device, R.real.dtype)
        Vw = Vmat * weights[:, None]
        Rw = R_work * weights
    else:
        Vw = Vmat
        Rw = R_work
    sol = torch.linalg.lstsq(Vw, Rw).solution
    if append_bias:
        A = sol[:-1]
        bias = bias + sol[-1]
    else:
        A = sol

    # Fold extremely slow terms into the bias to avoid unstable near-zero exponents.
    if append_bias and A.numel() > 0:
        slow_mask = torch.abs(B.real) < 1e-6
        if torch.any(slow_mask):
            bias = bias + torch.sum(A[slow_mask])
            A = A[~slow_mask]
            B = B[~slow_mask]

    if A.numel() > 0:
        fitted = torch.exp(-B[None, :] * (k - k[0])[:, None]) @ A + bias
    else:
        fitted = torch.ones_like(R) * bias

    residual = torch.linalg.norm(fitted - R) / torch.linalg.norm(R).clamp_min(1e-12)
    diag = ExpFitDiagnostics(residual=float(residual.item()), model_order=A.numel(), eigvals=eigvals)
    return {"A": A, "B": B, "bias": bias, "diagnostics": diag}
