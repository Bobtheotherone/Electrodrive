import torch

from electrodrive.images.search import solve_l1_ista
from electrodrive.images.basis import BasisOperator
from electrodrive.utils.logging import JsonlLogger


def _make_logging_stub():
    class _Stub(JsonlLogger):
        def __init__(self):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    return _Stub()


def _build_operator_from_dense(A: torch.Tensor) -> BasisOperator:
    # Wrap columns as point bases that use precomputed columns for matvec/rmatvec.
    # For small synthetic tests we can implement a tiny adapter via BasisOperator
    # by directly setting .elements and overriding potential via lambda closures.
    class _ColumnBasis:
        def __init__(self, col: torch.Tensor):
            self.type = "dense_col"
            self.params = {}
            self._col = col

        def potential(self, targets: torch.Tensor) -> torch.Tensor:
            # targets are ignored; we return the precomputed column aligned to len(targets)
            # This keeps matvec = column * weight when targets match A rows.
            return self._col.to(device=targets.device, dtype=targets.dtype)

    elems = [_ColumnBasis(A[:, j]) for j in range(A.shape[1])]
    # BasisOperator expects points but only uses them for shape when matvec/rmatvec
    # are passed targets=None. Store dummy points sized to rows of A.
    dummy_points = torch.zeros(A.shape[0], 3, device=A.device, dtype=A.dtype)
    return BasisOperator(elems, points=dummy_points, device=A.device, dtype=A.dtype)


def _run_ista_dense_and_op(A: torch.Tensor, w_true: torch.Tensor):
    y = A @ w_true
    logger = _make_logging_stub()
    # Dense path
    w_dense, _ = solve_l1_ista(
        A,
        y,
        reg_l1=1e-3,
        logger=logger,
        max_iter=200,
        tol=1e-6,
        normalize_columns=True,
    )
    # Operator path
    op = _build_operator_from_dense(A)
    w_op, _ = solve_l1_ista(
        op,
        y,
        reg_l1=1e-3,
        logger=logger,
        max_iter=200,
        tol=1e-6,
        normalize_columns=True,
    )
    return w_dense, w_op


def test_operator_parity_normalized():
    torch.manual_seed(0)
    N, K = 64, 8
    A = torch.randn(N, K)
    w_true = torch.zeros(K)
    w_true[[1, 4]] = torch.tensor([0.8, -1.2])
    w_dense, w_op = _run_ista_dense_and_op(A, w_true)
    rel_err = torch.linalg.norm(w_dense - w_op) / torch.linalg.norm(w_dense + 1e-9)
    assert rel_err < 1e-2


def test_operator_handles_ill_scaled_columns():
    torch.manual_seed(1)
    N, K = 64, 8
    A = torch.randn(N, K)
    scale = torch.ones(K)
    scale[:3] = 1e6
    A_bad = A * scale
    w_true = torch.zeros(K)
    w_true[0] = 1.5
    w_true[4] = -0.7
    w_dense, w_op = _run_ista_dense_and_op(A_bad, w_true)
    rel_err = torch.linalg.norm(w_dense - w_op) / torch.linalg.norm(w_dense + 1e-9)
    assert rel_err < 1e-2
    # Check recovered predictions remain close to ground truth despite ill-scaling.
    y_true = A_bad @ w_true
    pred_dense = A_bad @ w_dense
    pred_op = A_bad @ w_op
    assert torch.allclose(pred_dense, y_true, atol=1e-2, rtol=1e-2)
    assert torch.allclose(pred_op, y_true, atol=1e-2, rtol=1e-2)


def test_operator_zero_columns_safe():
    torch.manual_seed(2)
    N, K = 32, 6
    A = torch.randn(N, K)
    A[:, 2] = 0.0  # zero column
    w_true = torch.zeros(K)
    w_true[1] = 0.5
    y = A @ w_true

    op = _build_operator_from_dense(A)
    logger = _make_logging_stub()
    w_op, _ = solve_l1_ista(
        op,
        y,
        reg_l1=1e-3,
        logger=logger,
        max_iter=100,
        tol=1e-6,
        normalize_columns=True,
    )
    pred = A @ w_op
    assert torch.isfinite(pred).all()
    assert torch.allclose(pred, y, atol=1e-3, rtol=1e-3)
