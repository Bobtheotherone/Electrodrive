import torch
import pytest

from electrodrive.images.search import solve_sparse
from electrodrive.utils.logging import JsonlLogger


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for implicit solver grad flow")
def test_solve_sparse_backward_grad_flow(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 64, 32
    A = torch.randn(n, k, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
    X = torch.empty((0, 3), device=device, dtype=dtype)

    with JsonlLogger(tmp_path) as logger:
        w, support, stats = solve_sparse(
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

    loss = (w ** 2).sum()
    loss.backward()

    assert A.grad is not None
    assert g.grad is not None
    assert torch.isfinite(A.grad).all()
    assert torch.isfinite(g.grad).all()
    assert isinstance(support, list)
    assert isinstance(stats, dict)
