import torch
import pytest

from electrodrive.images.basis import BasisOperator
from electrodrive.images.optim import DTypePolicy, SparseSolveRequest, implicit_lasso_solve


if not torch.cuda.is_available():
    pytest.skip("CUDA required for implicit solver tests", allow_module_level=True)


class _ColumnBasis:
    def __init__(self, col: torch.Tensor):
        self.type = "dense_col"
        self.params = {}
        self._col = col

    def potential(self, targets: torch.Tensor) -> torch.Tensor:
        return self._col.to(device=targets.device, dtype=targets.dtype)


def _build_operator_from_dense(A: torch.Tensor) -> BasisOperator:
    elems = [_ColumnBasis(A[:, j]) for j in range(A.shape[1])]
    dummy_points = torch.zeros(A.shape[0], 3, device=A.device, dtype=A.dtype)
    return BasisOperator(elems, points=dummy_points, device=A.device, dtype=A.dtype)


def _solve(A, g):
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
        dtype_policy=DTypePolicy(forward_dtype=g.dtype),
    )
    return implicit_lasso_solve(req).w


def test_dense_operator_parity():
    torch.manual_seed(2)
    device = torch.device("cuda")
    N, K = 32, 12
    A = torch.randn(N, K, device=device, dtype=torch.float32)
    w_true = torch.zeros(K, device=device, dtype=torch.float32)
    w_true[[2, 7]] = torch.tensor([1.1, -0.7], device=device)
    g = A @ w_true

    w_dense = _solve(A, g)
    op = _build_operator_from_dense(A)
    w_op = _solve(op, g)

    rel_err = torch.linalg.norm(w_dense - w_op) / (torch.linalg.norm(w_dense) + 1e-9)
    assert float(rel_err) < 2e-2
