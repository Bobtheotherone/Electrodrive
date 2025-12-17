from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from electrodrive.images.operator import BasisOperator
from .core import ConstraintSpec
from .bases.fourier_planar import PlanarFFTConstraintOp
from .bases.spherical_harmonics import SphericalHarmonicsConstraintOp
from .bases.fourier_bessel import CylindricalFourierConstraintOp


def _matvec(A: torch.Tensor | BasisOperator, w: torch.Tensor, X: Optional[torch.Tensor]) -> torch.Tensor:
    if isinstance(A, BasisOperator):
        pts = X if X is not None else getattr(A, "points", None)
        return A.matvec(w, pts)
    return A.matmul(w)


def _rmatvec(A: torch.Tensor | BasisOperator, r: torch.Tensor, X: Optional[torch.Tensor]) -> torch.Tensor:
    if isinstance(A, BasisOperator):
        pts = X if X is not None else getattr(A, "points", None)
        return A.rmatvec(r, pts)
    return A.transpose(0, 1).matmul(r)


def active_set_stats(w: torch.Tensor, threshold: float = 1e-6) -> dict[str, float | int]:
    w_abs = torch.abs(w)
    if w_abs.numel() == 0:
        return {"active_count": 0, "active_frac": 0.0}
    mask = w_abs > threshold
    active_count = int(mask.sum().item())
    active_frac = float(active_count) / float(w_abs.numel())
    return {"active_count": active_count, "active_frac": active_frac}


def conditioning_proxy(H: torch.Tensor) -> dict[str, float]:
    if H.numel() == 0:
        return {"diag_min": float("nan"), "diag_max": float("nan"), "diag_ratio": float("nan")}
    diag = torch.diag(H)
    diag_min = float(diag.min().item()) if diag.numel() else float("nan")
    diag_max = float(diag.max().item()) if diag.numel() else float("nan")
    diag_ratio = diag_max / max(diag_min, 1e-12) if diag_min == diag_min else float("nan")
    return {"diag_min": diag_min, "diag_max": diag_max, "diag_ratio": float(diag_ratio)}


def lasso_kkt_residual(
    A: torch.Tensor | BasisOperator,
    w: torch.Tensor,
    g: torch.Tensor,
    lambda_l1: float,
    *,
    weight_prior: Optional[torch.Tensor] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    X: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    if w.numel() == 0:
        return {"kkt_residual": float("nan"), "kkt_violation_max": float("nan")}
    r = _matvec(A, w, X) - g
    grad = _rmatvec(A, r, X)
    if weight_prior is not None and float(torch.as_tensor(lambda_weight_prior).max().item()) > 0.0:
        w_prior = weight_prior.to(device=w.device, dtype=w.dtype).view(-1)
        if w_prior.numel() == 1:
            w_prior = w_prior.expand_as(w)
        if w_prior.numel() != w.numel():
            w_prior = torch.nn.functional.pad(w_prior, (0, max(0, w.numel() - w_prior.numel())))[: w.numel()]
        lam = lambda_weight_prior
        if torch.is_tensor(lam):
            lam = lam.to(device=w.device, dtype=w.dtype).view(-1)
            if lam.numel() == 1:
                lam = lam.expand_as(w)
        grad = grad + (w - w_prior) * lam

    w_abs = torch.abs(w)
    sign = torch.sign(w)
    active = w_abs > 0.0
    residual = grad[active] + lambda_l1 * sign[active]
    violation = torch.clamp(grad[~active].abs() - lambda_l1, min=0.0) if bool((~active).any()) else torch.tensor(0.0, device=w.device)
    kkt_resid = float(torch.linalg.norm(residual).item()) if residual.numel() else 0.0
    kkt_violation = float(violation.max().item()) if violation.numel() else 0.0
    return {"kkt_residual": kkt_resid, "kkt_violation_max": kkt_violation}


class CudaTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.elapsed_ms: float = float("nan")
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        else:
            self._start = None
            self._end = None

    def __enter__(self) -> "CudaTimer":
        if self.enabled and self._start is not None:
            self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._start is not None and self._end is not None:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = float(self._start.elapsed_time(self._end))


@dataclass
class _ConstraintBlock:
    name: str
    kind: str
    eps: float
    op: Any


class WeightedConstraintOp:
    def __init__(self, op: Any, weight: float) -> None:
        self.op = op
        self.weight = float(weight)

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        return self.op.apply(r) * self.weight

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        return self.op.adjoint(c) * self.weight


class CollocationConstraintOp:
    def __init__(self, n_rows: int, indices: torch.Tensor, weights: Optional[torch.Tensor] = None) -> None:
        self.n_rows = int(n_rows)
        self.indices = indices.to(dtype=torch.long)
        self.weights = weights

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        r_sel = r.index_select(0, self.indices)
        if self.weights is not None:
            return r_sel * self.weights
        return r_sel

    def adjoint(self, c: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_rows, device=c.device, dtype=c.dtype)
        if self.weights is not None:
            out.index_add_(0, self.indices, c * self.weights)
        else:
            out.index_add_(0, self.indices, c)
        return out


def _constraint_violation(kind: str, c: torch.Tensor, eps: float) -> float:
    if kind == "eq":
        return float(torch.linalg.norm(c).item())
    if kind == "l2":
        norm = float(torch.linalg.norm(c).item())
        return max(0.0, norm - eps)
    if kind == "linf":
        max_abs = float(torch.max(torch.abs(c)).item()) if c.numel() else 0.0
        return max(0.0, max_abs - eps)
    return float("nan")


def _compile_constraints(
    specs: Sequence[ConstraintSpec],
    *,
    device: torch.device,
    dtype: torch.dtype,
    points: Optional[torch.Tensor],
    is_boundary: Optional[torch.Tensor],
    n_rows: int,
) -> list[_ConstraintBlock]:
    blocks: list[_ConstraintBlock] = []
    for idx, spec in enumerate(specs):
        basis = (spec.basis or "collocation").strip().lower()
        kind = (spec.kind or "eq").strip().lower()
        eps = float(spec.eps or 0.0)
        params = spec.params or {}
        name = spec.name or f"constraint_{idx}"

        if basis == "collocation":
            indices = None
            if "indices" in params:
                indices = torch.as_tensor(params["indices"], device=device, dtype=torch.long)
            elif "mask" in params:
                mask = torch.as_tensor(params["mask"], device=device, dtype=torch.bool)
                indices = torch.nonzero(mask, as_tuple=False).view(-1)
            elif spec.region in {"boundary", "interior"}:
                if is_boundary is None:
                    raise ValueError("Constraint region requested but is_boundary mask is missing.")
                mask = torch.as_tensor(is_boundary, device=device, dtype=torch.bool).view(-1)
                if spec.region == "interior":
                    mask = ~mask
                indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices is None:
                indices = torch.arange(n_rows, device=device, dtype=torch.long)
            weights = None
            if "weights" in params:
                weights = torch.as_tensor(params["weights"], device=device, dtype=dtype).view(-1)
            op = CollocationConstraintOp(n_rows=n_rows, indices=indices, weights=weights)
        elif basis in {"planar_fft", "fft_planar"}:
            grid_shape = params.get("grid_shape")
            if grid_shape is None:
                grid_h = params.get("grid_h")
                grid_w = params.get("grid_w")
                if grid_h is None or grid_w is None:
                    raise ValueError("Planar FFT constraint requires grid_shape or grid_h/grid_w.")
                grid_shape = (int(grid_h), int(grid_w))
            mode_indices = params.get("mode_indices")
            mask = params.get("mask")
            fft_shift = bool(params.get("fft_shift", False))
            op = PlanarFFTConstraintOp(
                grid_shape=grid_shape,
                mode_indices=mode_indices,
                mask=mask,
                fft_shift=fft_shift,
                device=device,
                dtype=dtype,
            )
        elif basis in {"sphere_sh", "spherical_harmonics"}:
            lmax = params.get("Lmax")
            if lmax is None:
                lmax = params.get("lmax")
            if lmax is None:
                raise ValueError("Spherical harmonics constraint requires Lmax.")
            theta = params.get("theta")
            phi = params.get("phi")
            if theta is None or phi is None:
                if points is None:
                    raise ValueError("Spherical harmonics constraints need theta/phi or points.")
                pts = points.to(device=device, dtype=dtype)
                theta = torch.atan2(torch.linalg.norm(pts[:, :2], dim=1), pts[:, 2])
                phi = torch.atan2(pts[:, 1], pts[:, 0])
            op = SphericalHarmonicsConstraintOp(
                lmax=int(lmax),
                theta=theta,
                phi=phi,
                device=device,
                dtype=dtype,
            )
        elif basis in {"cylindrical", "fourier_bessel", "fourier_cyl"}:
            n_phi = int(params.get("n_phi", 16))
            n_z = int(params.get("n_z", 8))
            phi = params.get("phi")
            z = params.get("z")
            if phi is None or z is None:
                if points is None:
                    raise ValueError("Cylindrical constraints need phi/z or points.")
                pts = points.to(device=device, dtype=dtype)
                phi = torch.atan2(pts[:, 1], pts[:, 0])
                z = pts[:, 2]
            op = CylindricalFourierConstraintOp(
                n_phi=n_phi,
                n_z=n_z,
                phi=phi,
                z=z,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported constraint basis: {basis}")

        if float(spec.weight) != 1.0:
            op = WeightedConstraintOp(op, float(spec.weight))

        blocks.append(_ConstraintBlock(name=name, kind=kind, eps=eps, op=op))
    return blocks


def constraint_residuals_from_specs(
    A: torch.Tensor | BasisOperator,
    w: torch.Tensor,
    g: torch.Tensor,
    constraints: Sequence[ConstraintSpec],
    *,
    X: Optional[torch.Tensor] = None,
    is_boundary: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    if not constraints:
        return {}
    r = _matvec(A, w, X) - g
    device = r.device
    dtype = r.dtype
    blocks = _compile_constraints(
        constraints,
        device=device,
        dtype=dtype,
        points=X,
        is_boundary=is_boundary,
        n_rows=r.shape[0],
    )
    residuals: dict[str, float] = {}
    for blk in blocks:
        c = blk.op.apply(r)
        residuals[blk.name] = _constraint_violation(blk.kind, c, blk.eps)
    return residuals



def residual_norms(
    A: torch.Tensor | BasisOperator,
    w: torch.Tensor,
    g: torch.Tensor,
    *,
    X: Optional[torch.Tensor] = None,
    is_boundary: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    resid = _matvec(A, w, X) - g
    stats: dict[str, float] = {}
    if is_boundary is not None and bool(is_boundary.any()):
        stats["boundary_residual"] = float(torch.linalg.norm(resid[is_boundary]).item())
        interior_mask = ~is_boundary
        stats["interior_residual"] = (
            float(torch.linalg.norm(resid[interior_mask]).item()) if bool(interior_mask.any()) else 0.0
        )
        stats["frac_boundary"] = float(is_boundary.float().mean().item())
    else:
        stats["boundary_residual"] = float(torch.linalg.norm(resid).item())
        stats["interior_residual"] = float(torch.linalg.norm(resid).item())
        stats["frac_boundary"] = 0.0
    return stats


def collect_solver_stats(
    *,
    solver: str,
    A: torch.Tensor | BasisOperator,
    w: torch.Tensor,
    g: torch.Tensor,
    lambda_l1: float,
    X: Optional[torch.Tensor] = None,
    is_boundary: Optional[torch.Tensor] = None,
    constraints: Optional[Sequence[ConstraintSpec]] = None,
    weight_prior: Optional[torch.Tensor] = None,
    lambda_weight_prior: float | torch.Tensor = 0.0,
    admm_stats: Optional[dict[str, Any]] = None,
    timing_ms: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "solver": solver,
        "dtype": str(w.dtype),
        "support_size": int((torch.abs(w) > 1e-6).sum().item()) if w.numel() else 0,
    }
    stats.update(active_set_stats(w))
    try:
        stats.update(
            lasso_kkt_residual(
                A,
                w,
                g,
                float(lambda_l1),
                weight_prior=weight_prior,
                lambda_weight_prior=lambda_weight_prior,
                X=X,
            )
        )
    except Exception:
        pass
    stats.update(residual_norms(A, w, g, X=X, is_boundary=is_boundary))
    if constraints:
        try:
            stats["constraint_residuals"] = constraint_residuals_from_specs(
                A,
                w,
                g,
                constraints,
                X=X,
                is_boundary=is_boundary,
            )
        except Exception:
            pass
    if admm_stats:
        stats["primal_res"] = admm_stats.get("primal_res", float("nan"))
        stats["dual_res"] = admm_stats.get("dual_res", float("nan"))
    if timing_ms:
        stats["gpu_ms"] = {k: float(v) for k, v in timing_ms.items()}
    return stats
