import torch
import pytest

from electrodrive.images.optim import DTypePolicy, SparseSolveRequest, implicit_grouplasso_solve


if not torch.cuda.is_available():
    pytest.skip("CUDA required for implicit solver tests", allow_module_level=True)


def _solve(A: torch.Tensor, g: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    req = SparseSolveRequest(
        A=A,
        X=None,
        g=g,
        is_boundary=None,
        lambda_l1=0.0,
        lambda_group=0.15,
        group_ids=group_ids,
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
    return implicit_grouplasso_solve(req).w


def test_implicit_grouplasso_grad_g():
    torch.manual_seed(1)
    device = torch.device("cuda")
    N, K = 32, 16
    A = torch.randn(N, K, device=device, dtype=torch.float32)
    group_ids = torch.arange(K, device=device, dtype=torch.long) // 4
    w_true = torch.zeros(K, device=device, dtype=torch.float32)
    w_true[0:4] = torch.tensor([0.7, -0.5, 0.4, 0.0], device=device)
    w_true[8:12] = torch.tensor([-0.6, 0.3, 0.0, 0.2], device=device)
    g = A @ w_true + 1e-2 * torch.randn(N, device=device)
    g.requires_grad_(True)

    v = torch.randn(K, device=device)
    w = _solve(A, g, group_ids)
    loss = torch.dot(w, v)
    loss.backward()

    grad_g = g.grad.detach().clone()
    eps = 1e-3
    for idx in [1, 6, 11]:
        g_plus = g.detach().clone()
        g_minus = g.detach().clone()
        g_plus[idx] += eps
        g_minus[idx] -= eps
        w_plus = _solve(A, g_plus, group_ids)
        w_minus = _solve(A, g_minus, group_ids)
        loss_plus = torch.dot(w_plus, v)
        loss_minus = torch.dot(w_minus, v)
        fd = (loss_plus - loss_minus) / (2.0 * eps)
        assert torch.allclose(grad_g[idx], fd, rtol=2e-2, atol=5e-3)
