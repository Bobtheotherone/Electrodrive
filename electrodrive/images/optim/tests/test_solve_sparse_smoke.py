import torch
import pytest

from electrodrive.images.search import solve_sparse
from electrodrive.utils.logging import JsonlLogger


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU-only solver smoke test")
def test_solve_sparse_smoke(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 64, 32
    A = torch.randn(n, k, device=device, dtype=dtype)
    w_true = torch.randn(k, device=device, dtype=dtype)
    g = A @ w_true + 0.01 * torch.randn(n, device=device, dtype=dtype)
    X = torch.empty((0, 3), device=device, dtype=dtype)

    with JsonlLogger(tmp_path) as logger:
        w_imp, support_imp, stats_imp = solve_sparse(
            A,
            X,
            g,
            None,
            logger,
            reg_l1=0.1,
            solver="implicit_lasso",
            max_iter=200,
            tol=1e-5,
            return_stats=True,
        )

        assert w_imp.shape == (k,)
        assert torch.isfinite(w_imp).all()
        assert isinstance(support_imp, list)
        assert all(isinstance(i, int) for i in support_imp)
        assert isinstance(stats_imp, dict)
        assert "solver" in stats_imp

        w_ista, support_ista, stats_ista = solve_sparse(
            A,
            X,
            g,
            None,
            logger,
            reg_l1=0.1,
            solver="ista",
            max_iter=50,
            tol=1e-5,
            return_stats=True,
        )

        assert w_ista.shape == (k,)
        assert torch.isfinite(w_ista).all()
        assert isinstance(support_ista, list)
        assert isinstance(stats_ista, dict)
        assert "solver" in stats_ista
