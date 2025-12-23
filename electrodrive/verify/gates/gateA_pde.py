from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from . import GateResult, _assert_cuda_inputs
from ..oracle_types import OracleQuery, OracleResult


def _ensure_vector(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError("candidate_eval must return a torch.Tensor or (V, E)")
    if not out.is_cuda:
        raise ValueError("candidate_eval must return CUDA tensors (GPU-first rule)")
    return out.flatten()


def _candidate_eval_fn(config: Dict[str, object]) -> Callable[[torch.Tensor], torch.Tensor]:
    fn = config.get("candidate_eval", None)
    if not callable(fn):
        raise ValueError("Gate A requires a callable 'candidate_eval' in config")
    return fn  # type: ignore[return-value]


def _bounds_from_spec(spec: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    charges = spec.get("charges", []) or []
    if charges:
        pts = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], dtype=torch.float32)
        lo = torch.min(pts, dim=0).values - 1.0
        hi = torch.max(pts, dim=0).values + 1.0
        return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))
    return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)


def _sample_interior(
    spec: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    *,
    seed: int,
    exclusion_radius: float,
) -> torch.Tensor:
    (x0, x1), (y0, y1), (z0, z1) = _bounds_from_spec(spec)
    # SobolEngine draws on CPU; transfer immediately to CUDA to keep hot paths on GPU.
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    pts = engine.draw(n).to(device=device, dtype=dtype)
    span = torch.tensor([x1 - x0, y1 - y0, z1 - z0], device=device, dtype=dtype)
    base = torch.tensor([x0, y0, z0], device=device, dtype=dtype)
    pts = pts * span + base

    charges = spec.get("charges", []) or []
    if charges:
        charge_pos = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], device=device, dtype=dtype)
        dists = torch.cdist(pts, charge_pos)
        mask = torch.all(dists > exclusion_radius, dim=1)
        if torch.any(~mask):
            pts = pts[mask]
    if pts.numel() == 0:
        pts = torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0
    if not pts.is_cuda:
        raise ValueError("Gate A sampling produced CPU tensor (GPU-first rule)")
    return pts.contiguous()


def _laplacian_autograd(candidate_eval: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    pts = pts.detach().clone().requires_grad_(True)
    V = _ensure_vector(candidate_eval(pts))
    if V.shape[0] != pts.shape[0]:
        raise ValueError("candidate_eval must return one value per point")
    lap = torch.zeros_like(V)
    for idx in range(pts.shape[0]):
        grad_i = torch.autograd.grad(V[idx], pts, retain_graph=True, create_graph=True)[0][idx]
        second = 0.0
        for dim in range(3):
            second += torch.autograd.grad(grad_i[dim], pts, retain_graph=True)[0][idx, dim]
        lap[idx] = second
    return lap.detach()


def _laplacian_finite_diff(
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    pts: torch.Tensor,
    *,
    h: float,
) -> torch.Tensor:
    V0 = _ensure_vector(candidate_eval(pts))
    lap = torch.zeros_like(V0)
    eye = torch.eye(3, device=pts.device, dtype=pts.dtype) * h
    for dim in range(3):
        offset = eye[dim].unsqueeze(0)
        plus = _ensure_vector(candidate_eval(pts + offset))
        minus = _ensure_vector(candidate_eval(pts - offset))
        lap = lap + (plus - 2.0 * V0 + minus) / (h * h)
    return lap.detach()


def run_gate(
    query: OracleQuery,
    result: OracleResult,
    *,
    config: Optional[Dict[str, object]] = None,
) -> GateResult:
    _assert_cuda_inputs(query, result)
    cfg = dict(config or {})
    seed = int(cfg.get("seed", 0))
    n_samples = int(cfg.get("n_interior", 128))
    exclusion_radius = float(cfg.get("exclusion_radius", 5e-2))
    thresholds = {
        "linf": float(cfg.get("linf_tol", 5e-3)),
        "l2": float(cfg.get("l2_tol", 2e-3)),
        "p95": float(cfg.get("p95_tol", 3e-3)),
    }
    artifact_dir = cfg.get("artifact_dir", None)
    spec = dict(cfg.get("spec", query.spec))
    candidate_eval = _candidate_eval_fn(cfg)

    torch.manual_seed(seed)
    pts = _sample_interior(spec, query.points.device, query.points.dtype, n_samples, seed=seed, exclusion_radius=exclusion_radius)

    prefer_autograd = bool(cfg.get("prefer_autograd", False))
    lap_method = "finite_diff"
    fd_pts = pts
    try:
        if prefer_autograd and pts.shape[0] <= int(cfg.get("autograd_max_samples", 64)):
            lap = _laplacian_autograd(candidate_eval, pts)
            lap_method = "autograd"
        else:
            fd_pts = pts[: max(16, min(int(cfg.get("fd_max_samples", 128)), pts.shape[0]))].contiguous()
            lap = _laplacian_finite_diff(candidate_eval, fd_pts, h=float(cfg.get("fd_h", 2e-2)))
            pts = fd_pts
    except Exception:
        fd_pts = pts[: max(16, min(int(cfg.get("fd_max_samples", 128)), pts.shape[0]))].contiguous()
        lap = _laplacian_finite_diff(candidate_eval, fd_pts, h=float(cfg.get("fd_h", 2e-2)))
        lap_method = "finite_diff"
        pts = fd_pts

    abs_lap = lap.abs()
    linf = float(torch.max(abs_lap).item())
    l2 = float(torch.sqrt(torch.mean(abs_lap * abs_lap)).item())
    p95 = float(torch.quantile(abs_lap, 0.95).item())
    metrics = {"linf": linf, "l2": l2, "p95": p95, "n": float(pts.shape[0]), "method": 0.0}
    metrics["method"] = 0.0 if lap_method == "autograd" else 1.0

    status = "pass"
    if linf > thresholds["linf"] or l2 > thresholds["l2"]:
        status = "borderline" if linf <= thresholds["linf"] * 2.0 else "fail"

    evidence: Dict[str, str] = {}
    if artifact_dir:
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"points": pts.detach().cpu(), "laplacian": lap.detach().cpu()}, artifact_dir / "gateA_pde.pt")
            evidence["laplacian"] = str(artifact_dir / "gateA_pde.pt")
        except Exception:
            pass

    oracle_meta = {
        "method": result.method,
        "fidelity": result.fidelity.value,
        "config": result.config_fingerprint,
    }

    return GateResult(
        gate="A",
        status=status,
        metrics=metrics,
        thresholds=thresholds,
        evidence=evidence,
        oracle=oracle_meta,
        notes=[f"laplacian_method={lap_method}"],
        config=cfg,
    )
