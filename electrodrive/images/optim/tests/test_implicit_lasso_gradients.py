import torch
import pytest

from electrodrive.images.optim import DTypePolicy, SparseSolveRequest, implicit_lasso_solve


if not torch.cuda.is_available():
    pytest.skip("CUDA required for implicit solver tests", allow_module_level=True)


def _solve(A: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    req = SparseSolveRequest(
        A=A,
        X=None,
        g=g,
        is_boundary=None,
        lambda_l1=5e-2,
        lambda_group=0.0,
        group_ids=None,
        weight_prior=None,
        lambda_weight_prior=0.0,
        normalize_columns=True,
        col_norms=None,
        constraints=[],
        max_iter=300,
        tol=1e-7,
        warm_start=None,
        return_stats=False,
        dtype_policy=DTypePolicy(forward_dtype=A.dtype),
    )
    return implicit_lasso_solve(req).w


def test_implicit_lasso_grad_g():
    torch.manual_seed(0)
    device = torch.device("cuda")
    N, K = 32, 16
    A = torch.randn(N, K, device=device, dtype=torch.float32)
    w_true = torch.zeros(K, device=device, dtype=torch.float32)
    w_true[[1, 4, 7]] = torch.tensor([0.8, -1.2, 0.5], device=device)
    g = A @ w_true + 1e-2 * torch.randn(N, device=device)
    g.requires_grad_(True)

    v = torch.randn(K, device=device)
    w = _solve(A, g)
    loss = torch.dot(w, v)
    loss.backward()

    grad_g = g.grad.detach().clone()
    eps = 1e-3
    for idx in [0, 5, 10]:
        g_plus = g.detach().clone()
        g_minus = g.detach().clone()
        g_plus[idx] += eps
        g_minus[idx] -= eps
        w_plus = _solve(A, g_plus)
        w_minus = _solve(A, g_minus)
        loss_plus = torch.dot(w_plus, v)
        loss_minus = torch.dot(w_minus, v)
        fd = (loss_plus - loss_minus) / (2.0 * eps)
        assert torch.allclose(grad_g[idx], fd, rtol=2e-2, atol=5e-3)
