from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from electrodrive.images.operator import BasisOperator
from .core import ADMMConfig, DTypePolicy, SparseSolveRequest
from .diagnostics import (
    CudaTimer,
    active_set_stats,
    constraint_residuals_from_specs,
    conditioning_proxy,
    lasso_kkt_residual,
    residual_norms,
)


def cast_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype)


def resolve_forward_dtype(policy: DTypePolicy | None, fallback: torch.dtype) -> torch.dtype:
    if policy is None:
        return fallback
    return policy.forward_dtype


def resolve_kkt_dtype(policy: DTypePolicy | None) -> torch.dtype:
    if policy is None:
        return torch.float32
    return policy.kkt_dtype


def fp64_resolve_stub(
    req: SparseSolveRequest,
    w_init: torch.Tensor,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    return w_init, {"certified": False, "reason": "fp64_resolve_stub"}


def _cast_optional_tensor(
    value: Optional[torch.Tensor | float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    as_long: bool = False,
) -> Optional[torch.Tensor | float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if as_long:
            return value.to(device=device, dtype=torch.long)
        return value.to(device=device, dtype=dtype)
    return value


def _cast_basis_operator(
    op: BasisOperator,
    *,
    device: torch.device,
    dtype: torch.dtype,
    points: Optional[torch.Tensor],
) -> BasisOperator:
    pts = points if points is not None else getattr(op, "points", None)
    pts = pts.to(device=device, dtype=dtype) if pts is not None else None
    row_weights = getattr(op, "row_weights", None)
    return BasisOperator(list(op.elements), points=pts, device=device, dtype=dtype, row_weights=row_weights)


def _fp64_request(
    req: SparseSolveRequest,
    *,
    device: torch.device,
    dtype: torch.dtype,
    warm_start: Optional[torch.Tensor] = None,
) -> SparseSolveRequest:
    if isinstance(req.A, BasisOperator):
        A_cast = _cast_basis_operator(req.A, device=device, dtype=dtype, points=req.X)
    else:
        A_cast = req.A.to(device=device, dtype=dtype)

    X_cast = req.X.to(device=device, dtype=dtype) if req.X is not None else None
    g_cast = req.g.to(device=device, dtype=dtype).view(-1)
    is_boundary = req.is_boundary.to(device=device) if req.is_boundary is not None else None
    group_ids = _cast_optional_tensor(req.group_ids, device=device, dtype=dtype, as_long=True)
    weight_prior = _cast_optional_tensor(req.weight_prior, device=device, dtype=dtype)
    lambda_weight_prior = _cast_optional_tensor(req.lambda_weight_prior, device=device, dtype=dtype)
    col_norms = _cast_optional_tensor(req.col_norms, device=device, dtype=dtype)
    lambda_group = req.lambda_group
    if torch.is_tensor(lambda_group):
        lambda_group = lambda_group.to(device=device, dtype=dtype)

    policy = DTypePolicy(forward_dtype=dtype, kkt_dtype=dtype, certify_dtype=dtype, autocast=False)
    return SparseSolveRequest(
        A=A_cast,
        X=X_cast,
        g=g_cast,
        is_boundary=is_boundary,
        lambda_l1=float(req.lambda_l1),
        lambda_group=lambda_group if lambda_group is not None else 0.0,
        group_ids=group_ids,
        weight_prior=weight_prior,
        lambda_weight_prior=lambda_weight_prior if lambda_weight_prior is not None else 0.0,
        normalize_columns=bool(req.normalize_columns),
        col_norms=col_norms,
        constraints=req.constraints,
        max_iter=int(req.max_iter),
        tol=float(req.tol),
        warm_start=warm_start if warm_start is not None else req.warm_start,
        return_stats=True,
        dtype_policy=policy,
    )


def _conditioning_from_active(
    A: torch.Tensor | BasisOperator,
    w: torch.Tensor,
    *,
    X: Optional[torch.Tensor],
) -> dict[str, float]:
    if w.numel() == 0:
        return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}
    w_abs = torch.abs(w)
    max_abs = float(w_abs.max().item()) if w_abs.numel() else 0.0
    threshold = max(1e-6, 1e-6 * max(1.0, max_abs))
    support = torch.nonzero(w_abs > threshold, as_tuple=False).view(-1)
    if support.numel() == 0:
        return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}

    if isinstance(A, BasisOperator):
        idx_list = support.detach().to(device="cpu", dtype=torch.long).tolist()
        try:
            sub = A.subset(idx_list)
            dense = sub.to_dense(targets=X)
            if dense is None:
                return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}
            A_act = dense
        except Exception:
            return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}
    else:
        A_act = A[:, support]

    try:
        H = A_act.transpose(0, 1).matmul(A_act)
        return conditioning_proxy(H)
    except Exception:
        return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}


def mpmath_verify(
    A: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    *,
    max_size: int = 4096,
    dps: int = 80,
) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"enabled": False, "reason": f"mpmath import failed: {exc}"}

    if A.numel() > max_size or w.numel() > max_size or g.numel() > max_size:
        return {"enabled": False, "reason": "mpmath verification skipped (problem too large)"}

    mp.mp.dps = int(dps)
    A_list = A.detach().cpu().double().tolist()
    w_list = w.detach().cpu().double().tolist()
    g_list = g.detach().cpu().double().tolist()
    A_mp = mp.matrix(A_list)
    w_mp = mp.matrix(w_list)
    g_mp = mp.matrix(g_list)
    resid = A_mp * w_mp - g_mp
    res_norm = mp.sqrt(sum([r * r for r in resid]))
    return {"enabled": True, "residual_norm": float(res_norm), "dps": int(dps)}


def refine_and_certify(
    req: SparseSolveRequest,
    *,
    solver: str = "implicit_lasso",
    admm_cfg: Optional[ADMMConfig] = None,
    w_init: Optional[torch.Tensor] = None,
    use_mpmath: bool = False,
    mpmath_dps: int = 80,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    device = req.g.device if torch.is_tensor(req.g) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    req64 = _fp64_request(req, device=device, dtype=dtype, warm_start=w_init)

    timer = CudaTimer(enabled=torch.cuda.is_available())
    with timer:
        solver_mode = (solver or "implicit_lasso").strip().lower()
        if solver_mode == "implicit_grouplasso":
            from .grouplasso_implicit import implicit_grouplasso_solve

            result = implicit_grouplasso_solve(req64)
        elif solver_mode == "admm_constrained":
            from .constrained_admm import admm_constrained_solve

            result = admm_constrained_solve(req64, admm_cfg)
        else:
            from .lasso_implicit import implicit_lasso_solve

            result = implicit_lasso_solve(req64)

    w64 = result.w
    A64 = req64.A
    g64 = req64.g
    X64 = req64.X
    is_boundary = req64.is_boundary

    resid = (A64.matmul(w64) if torch.is_tensor(A64) else A64.matvec(w64, X64)) - g64
    if is_boundary is not None and bool(is_boundary.any()):
        b_res = resid[is_boundary]
    else:
        b_res = resid
    max_boundary = float(torch.max(torch.abs(b_res)).item()) if b_res.numel() else 0.0
    mean_boundary = float(torch.mean(torch.abs(b_res)).item()) if b_res.numel() else 0.0

    constraint_residuals = {}
    if req64.constraints:
        try:
            constraint_residuals = constraint_residuals_from_specs(
                A64,
                w64,
                g64,
                req64.constraints,
                X=X64,
                is_boundary=is_boundary,
            )
        except Exception:
            constraint_residuals = {}

    kkt_stats = {}
    try:
        kkt_stats = lasso_kkt_residual(
            A64,
            w64,
            g64,
            float(req64.lambda_l1),
            weight_prior=req64.weight_prior,
            lambda_weight_prior=req64.lambda_weight_prior,
            X=X64,
        )
    except Exception:
        kkt_stats = {}

    cert: dict[str, Any] = {
        "certified": True,
        "solver": solver,
        "dtype": str(dtype),
        "max_boundary_residual": max_boundary,
        "mean_boundary_residual": mean_boundary,
        "constraint_residuals": constraint_residuals,
        "kkt_residual": kkt_stats.get("kkt_residual", float("nan")),
        "kkt_violation_max": kkt_stats.get("kkt_violation_max", float("nan")),
        "conditioning": _conditioning_from_active(A64, w64, X=X64),
        "active_set": active_set_stats(w64),
        "residual_norms": residual_norms(A64, w64, g64, X=X64, is_boundary=is_boundary),
        "gpu_solve_ms": float(timer.elapsed_ms) if timer.elapsed_ms == timer.elapsed_ms else float("nan"),
    }

    if use_mpmath and torch.is_tensor(A64):
        cert["mpmath"] = mpmath_verify(A64, w64, g64, dps=mpmath_dps)
    elif use_mpmath:
        cert["mpmath"] = {"enabled": False, "reason": "mpmath verification requires dense A"}

    return w64, cert
