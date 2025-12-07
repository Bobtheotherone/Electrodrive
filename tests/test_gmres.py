from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from electrodrive.core.bem_solver import gmres_restart, setup_preconditioner


def _build_spd_system(
    n: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[torch.Tensor, callable, torch.Tensor, torch.Tensor]:
    """
    Build a small symmetric positive definite system A x = b
    with a known solution x_true.

    Returns
    -------
    A : (n, n) tensor
        SPD matrix.
    mv : callable
        Matvec closure v -> A @ v
    b : (n,) tensor
        Right-hand side.
    x_true : (n,) tensor
        Ground-truth solution such that A @ x_true = b.
    """
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    A = torch.randn(n, n, device=device)
    A = A.T @ A + 0.1 * torch.eye(n, device=device)
    x_true = torch.randn(n, device=device)
    b = A @ x_true

    def mv(v: torch.Tensor) -> torch.Tensor:
        return A @ v

    return A, mv, b, x_true


def _check_solution(
    x: torch.Tensor,
    x_true: torch.Tensor,
    b: torch.Tensor,
    mv,
    info: dict,
    tol: float = 1e-5,
) -> None:
    """Shared sanity checks on GMRES output.

    We primarily judge by numerical quality (residual, error), and treat
    the 'success' flag / 'code' as advisory. This keeps the tests robust
    if GMRES internal convergence criteria change slightly.
    """
    # Basic info contract
    assert "success" in info
    assert "resid" in info
    assert "iters" in info

    success = bool(info["success"])
    code = info.get("code", None)

    # Residual should be finite and small
    resid = float(info["resid"])
    assert math.isfinite(resid), f"GMRES resid is non-finite: {resid}"
    assert resid < tol, f"GMRES residual too large: resid={resid} (tol={tol})"

    # Solution error against ground truth
    err = torch.linalg.norm(x - x_true).item()
    assert err < tol, f"Solution error too large: err={err} (tol={tol})"

    # Double-check residual via matvec
    r = b - mv(x)
    r_norm = torch.linalg.norm(r).item()
    assert math.isfinite(r_norm), f"True residual is non-finite: r_norm={r_norm}"
    assert abs(r_norm - resid) < 10 * tol, (
        f"Reported residual {resid} inconsistent with true residual {r_norm}"
    )

    # Interpret status / code:
    # - success=True: great, nothing else to do.
    # - success=False, code==1: allow “near-miss” (e.g. hit maxiter) since
    #   residual and error are already tiny.
    # - success=False, any other code: treat as a real failure.
    if not success:
        if code is not None and code not in (0, 1):
            raise AssertionError(
                f"GMRES reported failure (success=False, code={code}) "
                f"despite small residual {resid}; this indicates a breakdown "
                "or numeric error."
            )


def test_gmres_converges_on_spd_system_cpu():
    A, mv, b, x_true = _build_spd_system(device="cpu")
    x, info = gmres_restart(A=mv, b=b, tol=1e-8, maxiter=200, restart=8)
    _check_solution(x, x_true, b, mv, info, tol=1e-5)


def test_gmres_with_diag_preconditioner_cpu():
    A, mv, b, x_true = _build_spd_system(device="cpu")
    M = setup_preconditioner("diag", A_diag=torch.diag(A))
    x, info = gmres_restart(A=mv, b=b, M=M, tol=1e-8, maxiter=200, restart=8)
    _check_solution(x, x_true, b, mv, info, tol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gmres_converges_on_spd_system_cuda():
    A, mv, b, x_true = _build_spd_system(device="cuda")
    x, info = gmres_restart(A=mv, b=b, tol=1e-8, maxiter=200, restart=8)
    _check_solution(x, x_true, b, mv, info, tol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gmres_with_diag_preconditioner_cuda():
    A, mv, b, x_true = _build_spd_system(device="cuda")
    M = setup_preconditioner("diag", A_diag=torch.diag(A))
    x, info = gmres_restart(A=mv, b=b, M=M, tol=1e-8, maxiter=200, restart=8)
    _check_solution(x, x_true, b, mv, info, tol=1e-4)
