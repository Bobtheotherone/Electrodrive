from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from . import GateResult, _assert_cuda_inputs
from ..oracle_types import OracleQuery, OracleResult


def _ensure_eval(cfg: Dict[str, object]) -> Callable[[torch.Tensor], torch.Tensor]:
    fn = cfg.get("candidate_eval", None)
    if not callable(fn):
        raise ValueError("Gate C requires 'candidate_eval' callable")
    return fn  # type: ignore[return-value]


def _fit_slope(r: torch.Tensor, vals: torch.Tensor) -> float:
    r = r.clamp_min(1e-6)
    vals = vals.clamp_min(1e-12)
    x = torch.log(r)
    y = torch.log(vals)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, y).solution
    return float(sol[0].item())


def _sample_on_sphere(n: int, radius: torch.Tensor) -> torch.Tensor:
    device = radius.device
    dtype = radius.dtype
    dirs = torch.randn(n, 3, device=device, dtype=dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(1e-6)
    return dirs * radius.unsqueeze(1)


def _sample_radial(
    n: int,
    r_min: float,
    r_max: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    r = torch.rand(n, device=device, dtype=dtype) * (r_max - r_min) + r_min
    pts = _sample_on_sphere(n, r)
    return pts, r


def run_gate(
    query: OracleQuery,
    result: OracleResult,
    *,
    config: Optional[Dict[str, object]] = None,
) -> GateResult:
    _assert_cuda_inputs(query, result)
    cfg = dict(config or {})
    candidate_eval = _ensure_eval(cfg)
    n_far = int(cfg.get("n_far", 96))
    n_near = int(cfg.get("n_near", 96))
    far_radius = float(cfg.get("far_radius", 10.0))
    near_radius = float(cfg.get("near_radius", 0.25))
    slope_tol = float(cfg.get("slope_tol", 0.15))
    spurious_tol = float(cfg.get("spurious_tol", 1e3))
    seed = int(cfg.get("seed", 0))

    device = query.points.device
    dtype = query.points.dtype

    far_pts, far_r = _sample_radial(n_far, far_radius, far_radius * 2.0, device, dtype, seed=seed)
    near_pts, near_r = _sample_radial(n_near, near_radius * 0.25, near_radius, device, dtype, seed=seed + 1)

    far_vals = candidate_eval(far_pts)
    if isinstance(far_vals, tuple):
        far_vals = far_vals[0]
    if not torch.is_tensor(far_vals) or not far_vals.is_cuda:
        raise ValueError("Gate C expects CUDA tensor outputs from candidate_eval")
    far_abs = torch.abs(far_vals.flatten())
    far_slope = _fit_slope(far_r, far_abs)

    near_vals = candidate_eval(near_pts)
    if isinstance(near_vals, tuple):
        near_vals = near_vals[0]
    if not torch.is_tensor(near_vals) or not near_vals.is_cuda:
        raise ValueError("Gate C expects CUDA tensor outputs from candidate_eval")
    near_abs = torch.abs(near_vals.flatten())
    near_slope = _fit_slope(near_r, near_abs)

    spurious_mask = far_abs > spurious_tol
    spurious_fraction = float(torch.mean(spurious_mask.float()).item())

    metrics = {
        "far_slope": far_slope,
        "near_slope": near_slope,
        "spurious_fraction": spurious_fraction,
        "far_radius": far_radius,
        "near_radius": near_radius,
    }
    thresholds = {"slope_tol": slope_tol, "spurious": spurious_tol}

    status = "pass"
    expected_slope = -1.0
    if abs(far_slope - expected_slope) > slope_tol or abs(near_slope - expected_slope) > slope_tol:
        status = "borderline"
    if spurious_fraction > 0.05:
        status = "fail"

    evidence = {}
    if "artifact_dir" in cfg and cfg["artifact_dir"]:
        path = cfg["artifact_dir"] / "gateC_asymptotics.pt"  # type: ignore[operator]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "far_points": far_pts.detach().cpu(),
                    "far_vals": far_vals.detach().cpu(),
                    "near_points": near_pts.detach().cpu(),
                    "near_vals": near_vals.detach().cpu(),
                },
                path,
            )
            evidence["samples"] = str(path)
        except Exception:
            pass

    oracle_meta = {"method": result.method, "fidelity": result.fidelity.value}
    return GateResult(
        gate="C",
        status=status,
        metrics=metrics,
        thresholds=thresholds,
        evidence=evidence,
        oracle=oracle_meta,
        notes=[],
        config=cfg,
    )
