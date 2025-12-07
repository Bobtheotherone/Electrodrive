from __future__ import annotations

import torch

from electrodrive.images.basis import PointChargeBasis, build_dictionary
from electrodrive.images.learned_solver import LISTALayer
from electrodrive.images.operator import BasisOperator
from electrodrive.images.search import solve_l1_ista


class DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def _make_toy_basis(dtype: torch.dtype) -> list[PointChargeBasis]:
    return [
        PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.5], dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([0.25, -0.1, 1.1], dtype=dtype)}),
    ]


def test_basis_operator_matches_dense_matvec_and_rmatvec() -> None:
    dtype = torch.float64
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.2],
            [0.1, -0.2, 0.6],
            [0.5, 0.5, 1.0],
        ],
        dtype=dtype,
    )
    basis = _make_toy_basis(dtype)

    Phi = build_dictionary(basis, pts, dtype=dtype)
    row_weights = torch.tensor([1.0, 0.25, 2.0], dtype=dtype)
    op = BasisOperator(basis, pts, dtype=dtype, row_weights=row_weights)

    w = torch.tensor([0.4, -1.2], dtype=dtype)
    r = torch.tensor([-0.5, 0.7, 0.25], dtype=dtype)

    Phi_weighted = Phi * torch.sqrt(row_weights).view(-1, 1)

    mat_dense = Phi_weighted @ w
    mat_op = op.matvec(w, pts)
    assert torch.allclose(mat_op, mat_dense, atol=1e-10, rtol=1e-7)

    rmat_dense = Phi_weighted.T @ r
    rmat_op = op.rmatvec(r, pts)
    assert torch.allclose(rmat_op, rmat_dense, atol=1e-10, rtol=1e-7)


def test_ista_accepts_operator_and_matches_dense() -> None:
    dtype = torch.float64
    pts = torch.tensor(
        [
            [-0.2, 0.0, 0.4],
            [0.0, 0.0, 0.6],
            [0.2, 0.1, 0.8],
            [0.3, -0.2, 1.1],
        ],
        dtype=dtype,
    )
    basis = _make_toy_basis(dtype)

    Phi = build_dictionary(basis, pts, dtype=dtype)
    op = BasisOperator(basis, pts, dtype=dtype)

    true_w = torch.tensor([0.75, -0.35], dtype=dtype)
    g = Phi @ true_w

    logger = DummyLogger()

    w_dense, _ = solve_l1_ista(Phi, g, reg_l1=1e-6, logger=logger, max_iter=200)
    w_op, _ = solve_l1_ista(op, g, reg_l1=1e-6, logger=logger, max_iter=200, collocation=pts)

    assert torch.allclose(w_dense, true_w, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_op, true_w, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_dense, w_op, atol=5e-4, rtol=5e-4)


def test_operator_ista_respects_row_weights_matches_dense() -> None:
    """Operator-mode ISTA with row weights + column norms matches dense path."""
    dtype = torch.float64
    pts = torch.stack(
        [
            torch.linspace(-0.3, 0.3, 4),
            torch.linspace(-0.2, 0.2, 4),
            torch.linspace(0.4, 0.8, 4),
        ],
        dim=1,
    )  # 4 points
    # Duplicate to make a slightly larger set with mixed boundary/interior.
    pts = torch.cat([pts, pts + torch.tensor([0.05, -0.05, 0.05])], dim=0)  # [8,3]

    boundary_mask = torch.zeros(pts.shape[0], dtype=torch.bool)
    boundary_mask[:4] = True  # first half boundary
    row_weights = torch.where(
        boundary_mask,
        torch.full_like(boundary_mask, 4.0, dtype=dtype),
        torch.ones_like(boundary_mask, dtype=dtype),
    )

    basis = [
        PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.5], dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([0.15, -0.1, 0.7], dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([-0.1, 0.2, 0.9], dtype=dtype)}),
    ]

    Phi = build_dictionary(basis, pts, dtype=dtype)
    rw_sqrt = torch.sqrt(row_weights).view(-1, 1)
    A_dense = Phi * rw_sqrt

    op = BasisOperator(basis, pts, dtype=dtype, row_weights=row_weights)

    w_true = torch.tensor([0.6, -0.3, 0.1], dtype=dtype)
    g = A_dense @ w_true

    logger = DummyLogger()
    reg = 1e-6

    w_dense, _ = solve_l1_ista(A_dense, g, reg_l1=reg, logger=logger, max_iter=1500)
    w_op, _ = solve_l1_ista(op, g, reg_l1=reg, logger=logger, max_iter=1500, collocation=pts)

    assert torch.allclose(w_dense, w_op, atol=5e-4, rtol=5e-4)

    support_dense = torch.where(torch.abs(w_dense) > 1e-4)[0].tolist()
    support_op = torch.where(torch.abs(w_op) > 1e-4)[0].tolist()
    assert support_dense == support_op


def test_lista_identity_and_group_prox() -> None:
    """LISTA layer matches identity solve and respects group sparsity prox."""
    dtype = torch.float32
    A = torch.eye(3, dtype=dtype)
    g = torch.tensor([1.0, -0.5, 0.25], dtype=dtype)

    lista = LISTALayer(K=3, n_steps=3, dense_threshold=8, init_theta=1e-6)
    w_plain = lista(A, None, g)
    assert torch.allclose(w_plain, g, atol=1e-4, rtol=1e-3)

    group_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    w_group = lista(A, None, g, group_ids=group_ids, lambda_group=0.4)
    expected = torch.tensor([0.643, -0.3215, 0.0], dtype=dtype)
    assert torch.allclose(w_group, expected, atol=5e-3, rtol=1e-2)


def test_lista_operator_matches_ista() -> None:
    """Operator-mode LISTA tracks ISTA on a toy basis."""
    dtype = torch.float32
    pts = torch.tensor(
        [
            [-0.2, 0.0, 0.4],
            [0.0, 0.0, 0.6],
            [0.2, 0.1, 0.8],
            [0.3, -0.2, 1.1],
        ],
        dtype=dtype,
    )
    basis = [
        PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.5], dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([0.25, -0.1, 1.1], dtype=dtype)}),
        PointChargeBasis({"position": torch.tensor([-0.15, 0.05, 0.9], dtype=dtype)}),
    ]

    op = BasisOperator(basis, pts, dtype=dtype)
    w_true = torch.tensor([0.8, -0.4, 0.2], dtype=dtype)
    g = op.matvec(w_true, pts)

    logger = DummyLogger()
    w_ista, _ = solve_l1_ista(op, g, reg_l1=1e-6, logger=logger, max_iter=300, collocation=pts)

    lista = LISTALayer(K=len(basis), n_steps=8, init_theta=1e-6, init_L=10.0)
    w_lista = lista(op, pts, g)

    assert torch.isfinite(w_lista).all()
    assert torch.max(torch.abs(w_lista - w_ista)).item() < 0.6
