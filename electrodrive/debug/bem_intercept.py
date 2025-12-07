"""
Lightweight live-intercept helper for BEM + collocation flows.

Controlled entirely by environment variables:
- EDE_BEM_INTERCEPT_MODE   : "short" | "deep"
- EDE_BEM_INTERCEPT_GEOM   : geometry label to match (e.g., "plane")
- EDE_BEM_INTERCEPT_OUTDIR : output directory (default experiments/_agent_outputs)
- EDE_BEM_INTERCEPT_RUN_ID : optional fixed run id
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from electrodrive.utils.config import EPS_0


_CTX_CACHE: Dict[str, "InterceptContext"] = {}


def _now_run_id(geom: str, mode: str) -> str:
    ts = int(time.time())
    return f"{geom}_{mode}_{ts}"


def _safe_json_write(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    except Exception:
        pass


def _load_existing(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _geom_from_spec(spec: Any) -> str:
    ctypes = sorted({c.get("type") for c in getattr(spec, "conductors", [])})
    if ctypes == ["plane"]:
        if len(spec.conductors) == 1:
            return "plane"
        if len(spec.conductors) == 2:
            return "parallel_planes"
    if ctypes == ["sphere"] and len(spec.conductors) == 1:
        return "sphere"
    return "unknown"


def _snapshot_bem_cfg(cfg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
        except TypeError:
            continue
        out[k] = v
    return out


@dataclass
class InterceptContext:
    enabled: bool
    mode: str
    geom: str
    run_id: str
    out_path: Path
    skip_heavy: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    collocation_buffer: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _parse_env() -> Optional[Dict[str, Any]]:
    mode = (os.getenv("EDE_BEM_INTERCEPT_MODE") or "").strip().lower()
    geom = (os.getenv("EDE_BEM_INTERCEPT_GEOM") or "").strip().lower()
    if mode not in {"short", "deep"}:
        return None
    if not geom:
        return None
    outdir = Path(os.getenv("EDE_BEM_INTERCEPT_OUTDIR") or "experiments/_agent_outputs")
    run_id_env = os.getenv("EDE_BEM_INTERCEPT_RUN_ID")
    return {"mode": mode, "geom": geom, "outdir": outdir, "run_id_env": run_id_env}


def maybe_start_intercept(
    spec: Any,
    test_name: Optional[str] = None,
    bem_cfg: Optional[Any] = None,
) -> Optional[InterceptContext]:
    env = _parse_env()
    if env is None:
        return None

    geom = _geom_from_spec(spec)
    if geom != env["geom"]:
        return None

    mode = env["mode"]
    run_id = env["run_id_env"] or _now_run_id(geom, mode)
    out_path = Path(env["outdir"]) / f"{run_id}.json"

    if run_id in _CTX_CACHE:
        ctx = _CTX_CACHE[run_id]
        if test_name:
            ctx.payload.setdefault("test", test_name)
        if bem_cfg is not None:
            ctx.payload.setdefault("bem_config", _snapshot_bem_cfg(bem_cfg))
        return ctx

    existing = _load_existing(out_path)
    skip_heavy = bool(mode == "deep" and existing is not None)

    payload: Dict[str, Any] = existing or {}
    payload.setdefault("run_id", run_id)
    payload.setdefault("geom", geom)
    payload.setdefault("mode", mode)
    if test_name:
        payload.setdefault("test", test_name)
    if bem_cfg is not None:
        payload.setdefault("bem_config", _snapshot_bem_cfg(bem_cfg))
    payload.setdefault("refinement_passes", [])

    ctx = InterceptContext(
        enabled=True,
        mode=mode,
        geom=geom,
        run_id=run_id,
        out_path=out_path,
        skip_heavy=skip_heavy,
        payload=payload,
    )
    _CTX_CACHE[run_id] = ctx
    return ctx


def record_bem_pass(ctx: InterceptContext, pass_payload: Dict[str, Any]) -> None:
    if not ctx or not ctx.enabled:
        return
    ctx.payload.setdefault("refinement_passes", []).append(pass_payload)


def set_stop_reason(ctx: InterceptContext, reason: str) -> None:
    if not ctx or not ctx.enabled:
        return
    ctx.payload["stop_reason"] = reason


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def attach_collocation(
    ctx: InterceptContext,
    supervision_mode: str,
    geom_type: str,
    points: torch.Tensor,
    V_gt: torch.Tensor,
    is_boundary: torch.Tensor,
    mask_finite: torch.Tensor,
    *,
    ratio_boundary: float,
    needs_eps_scaling: bool,
) -> None:
    if not ctx or not ctx.enabled:
        return
    if geom_type != ctx.geom:
        return

    pts_np = points.detach().cpu().numpy()
    V_np = V_gt.detach().cpu().numpy()
    bnd_np = is_boundary.detach().cpu().numpy().astype(bool)
    mask_np = mask_finite.detach().cpu().numpy().astype(bool)

    ctx.collocation_buffer[supervision_mode] = {
        "points": pts_np,
        "V": V_np,
        "is_boundary": bnd_np,
        "mask": mask_np,
        "ratio_boundary": float(ratio_boundary),
    }

    # Need both analytic and bem to compare.
    if not {"analytic", "bem"} <= set(ctx.collocation_buffer.keys()):
        return

    Va = ctx.collocation_buffer["analytic"]
    Vb = ctx.collocation_buffer["bem"]
    if Va["points"].shape != Vb["points"].shape:
        ctx.payload["collocation"] = {
            "n_points": int(Va["points"].shape[0]),
            "ratio_boundary": float(Va["ratio_boundary"]),
            "note": "point shape mismatch",
        }
        return

    mask_joint = Va["mask"] & Vb["mask"]

    Va_vals = Va["V"]
    Vb_vals = Vb["V"] * (EPS_0 if needs_eps_scaling else 1.0)
    bnd_mask = Va["is_boundary"] & Vb["is_boundary"] & mask_joint
    int_mask = (~Va["is_boundary"]) & (~Vb["is_boundary"]) & mask_joint
    denom = np.abs(Va_vals) + np.abs(Vb_vals) + 1e-9
    rel_full = np.abs(Va_vals - Vb_vals) / denom
    rel_overall_max = float(np.nanmax(rel_full[mask_joint])) if mask_joint.any() else float("nan")
    rel_bnd_max = float(np.nanmax(rel_full[bnd_mask])) if bnd_mask.any() else float("nan")
    rel_int_max = float(np.nanmax(rel_full[int_mask])) if int_mask.any() else float("nan")

    colloc_payload: Dict[str, Any] = {
        "n_points": int(Va_vals.shape[0]),
        "ratio_boundary": float(Va["ratio_boundary"]),
        "mask_finite_frac": float(mask_joint.mean()),
        "rel_err_overall_max": rel_overall_max,
        "rel_err_boundary_max": rel_bnd_max,
        "rel_err_interior_max": rel_int_max,
        "Va_stats": _stats(Va_vals[mask_joint]),
        "Vb_stats": _stats(Vb_vals[mask_joint]),
    }

    if ctx.mode == "deep" and not ctx.skip_heavy:
        n_samp = min(16, Va_vals.shape[0])
        idx = np.linspace(0, Va_vals.shape[0] - 1, n_samp, dtype=int)
        samples = []
        for i in idx:
            samples.append(
                {
                    "idx": int(i),
                    "point": [float(x) for x in pts_np[i]],
                    "is_boundary": bool(bnd_np[i]),
                    "Va": float(Va_vals[i]),
                    "Vb_scaled": float(Vb_vals[i]),
                    "rel_err": float(rel_full[i]),
                }
            )
        colloc_payload["samples"] = samples

    ctx.payload["collocation"] = colloc_payload


def finalize(ctx: Optional[InterceptContext]) -> None:
    if ctx is None or not ctx.enabled:
        return
    _safe_json_write(ctx.out_path, ctx.payload)
