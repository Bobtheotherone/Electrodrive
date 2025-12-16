import math

import pytest
import torch

from electrodrive.spectral.vector_fitting import vector_fit
from electrodrive.spectral.exp_fit import exp_fit


def _make_rational(k: torch.Tensor, poles: torch.Tensor, residues: torch.Tensor, d: complex, h: complex) -> torch.Tensor:
    V = 1.0 / (k[:, None] - poles[None, :])
    return torch.sum(residues[None, :] * V, dim=1) + d + h * k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_vector_fit_recovers_synthetic_poles():
    device = torch.device("cuda")
    dtype = torch.complex128
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    k = torch.linspace(0.1, 5.0, 256, device=device, dtype=real_dtype)
    true_poles = torch.tensor([-1.0 + 2.0j, -1.0 - 2.0j, -3.0 + 0.0j], device=device, dtype=dtype)
    true_residues = torch.tensor([0.6 - 0.1j, 0.6 + 0.1j, 0.3 + 0.05j], device=device, dtype=dtype)
    d = torch.tensor(0.1 + 0.05j, device=device, dtype=dtype)
    h = torch.tensor(0.02j, device=device, dtype=dtype)
    F = _make_rational(k, true_poles, true_residues, d, h)

    out = vector_fit(k, F, M=3, max_iters=10, tol=1e-8, device=device, dtype=dtype)
    poles_hat = out["poles"]
    residues_hat = out["residues"]
    assert poles_hat.device.type == "cuda"

    # Match estimated poles to true poles by nearest neighbour.
    for p_true in true_poles:
        idx = int(torch.argmin(torch.abs(poles_hat - p_true)).item())
        assert torch.abs(poles_hat[idx] - p_true) < 1e-2
        r_conj = residues_hat[idx]
        partner_idx = int(torch.argmin(torch.abs(poles_hat - p_true.conj())).item())
        if partner_idx != idx:
            assert torch.allclose(residues_hat[partner_idx], torch.conj(r_conj), atol=5e-2, rtol=5e-2)

    # Stable and conjugate symmetric.
    assert torch.all(poles_hat.real < -1e-5)
    n_pos = int(torch.sum(poles_hat.imag > 1e-6).item())
    n_neg = int(torch.sum(poles_hat.imag < -1e-6).item())
    assert n_pos == n_neg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_exp_fit_recovers_exponentials():
    device = torch.device("cuda")
    dtype = torch.complex128
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    k = torch.linspace(0.0, 1.0, 128, device=device, dtype=real_dtype)
    A_true = torch.tensor([1.0 + 0.2j, 0.6 - 0.1j], device=device, dtype=dtype)
    B_true = torch.tensor([0.5 + 0.1j, 1.1 + 0.2j], device=device, dtype=dtype)
    R = torch.sum(A_true[None, :] * torch.exp(-B_true[None, :] * k[:, None]), dim=1)
    R = R + 1e-4 * (torch.randn_like(R.real) + 1j * torch.randn_like(R.real))

    out = exp_fit(k, R, n_terms=2, device=device, dtype=dtype)
    B_hat = out["B"]
    bias = out.get("bias", torch.tensor(0.0, device=device, dtype=dtype))
    # Sort by real part to match order.
    idx_true = torch.argsort(B_true.real)
    idx_hat = torch.argsort(B_hat.real)
    B_true_sorted = B_true[idx_true]
    B_hat_sorted = B_hat[idx_hat]
    assert torch.all(B_hat_sorted.real > 0)
    rel_errs = torch.abs(B_hat_sorted.real - B_true_sorted.real) / B_true_sorted.real.clamp_min(1e-3)
    assert torch.all(rel_errs < 0.2)
    assert torch.is_tensor(bias)
