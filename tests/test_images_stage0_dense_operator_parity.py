from __future__ import annotations

import numpy as np
import torch

from electrodrive.images.basis import BasisOperator, generate_candidate_basis
from electrodrive.images.search import assemble_basis_matrix, get_collocation_data, solve_l1_ista
from electrodrive.orchestration.spec_registry import load_stage0_sphere_external


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def test_stage0_sphere_dense_operator_equivalence() -> None:
    """Dense matrix ISTA matches operator ISTA on a fixed Stage-0 sphere setup."""
    device = torch.device("cpu")
    dtype = torch.float64
    logger = _NullLogger()
    rng = np.random.default_rng(0)

    spec = load_stage0_sphere_external()
    colloc_pts, target, is_boundary = get_collocation_data(
        spec,
        logger,
        device=device,
        dtype=dtype,
        return_is_boundary=True,
        rng=rng,
        n_points_override=96,
        ratio_override=0.5,
    )
    assert colloc_pts.numel() > 0

    candidates = generate_candidate_basis(
        spec,
        basis_types=["sphere_kelvin_ladder", "axis_point", "point"],
        n_candidates=12,
        device=device,
        dtype=dtype,
    )
    A_dense = assemble_basis_matrix(candidates, colloc_pts)
    op = BasisOperator(candidates, colloc_pts, device=device, dtype=dtype)

    reg = 1e-6
    w_dense, _ = solve_l1_ista(
        A_dense,
        target,
        reg_l1=reg,
        logger=logger,
        max_iter=600,
        tol=1e-8,
        is_boundary=is_boundary,
    )
    w_op, _ = solve_l1_ista(
        op,
        target,
        reg_l1=reg,
        logger=logger,
        max_iter=600,
        tol=1e-8,
        collocation=colloc_pts,
        is_boundary=is_boundary,
    )

    rel_w = torch.linalg.norm(w_dense - w_op) / max(torch.linalg.norm(w_dense), torch.tensor(1e-6, dtype=dtype))
    assert float(rel_w) < 1e-2

    pred_dense = A_dense @ w_dense
    pred_op = op.matvec(w_op, colloc_pts)
    rel_pred = torch.linalg.norm(pred_dense - pred_op) / max(torch.linalg.norm(pred_dense), torch.tensor(1e-6, dtype=dtype))
    assert float(rel_pred) < 1e-3
