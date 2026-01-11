from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from electrodrive.orchestration.parser import CanonicalSpec


def _tensor_absmax(t: Optional[torch.Tensor]) -> float:
    if t is None or t.numel() == 0:
        return float("nan")
    flat = t.reshape(-1)
    finite = torch.isfinite(flat)
    if not torch.any(finite):
        return float("nan")
    vals = torch.abs(flat[finite])
    return float(torch.max(vals).item())


def _nonfinite_count(t: torch.Tensor) -> int:
    if t.numel() == 0:
        return 0
    return int(torch.count_nonzero(~torch.isfinite(t)).item())


def _charge_positions(
    spec: CanonicalSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    charges = getattr(spec, "charges", None)
    if isinstance(spec, dict):
        charges = spec.get("charges", [])
    charges = charges or []
    positions: List[List[float]] = []
    for ch in charges:
        if not isinstance(ch, dict):
            continue
        if ch.get("type") not in (None, "point"):
            continue
        pos = ch.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if not positions:
        return None
    return torch.tensor(positions, device=device, dtype=dtype)


def _min_distance_to_charges(
    pts: torch.Tensor,
    charge_pos: Optional[torch.Tensor],
) -> float:
    if charge_pos is None or pts.numel() == 0:
        return float("nan")
    dists = torch.cdist(pts, charge_pos)
    if dists.numel() == 0:
        return float("nan")
    return float(torch.min(dists).item())


def _scalar_to_json(val: torch.Tensor) -> Any:
    if torch.is_complex(val):
        return {"real": float(val.real.item()), "imag": float(val.imag.item())}
    return float(val.item())


def _topk_abs_entries(
    V_train: torch.Tensor,
    X_train: torch.Tensor,
    *,
    k: int,
) -> List[Dict[str, Any]]:
    if V_train.numel() == 0 or X_train.numel() == 0:
        return []
    flat_v = V_train.reshape(-1)
    if X_train.shape[0] != flat_v.shape[0]:
        return []
    abs_vals = torch.abs(flat_v)
    finite = torch.isfinite(abs_vals)
    if not torch.all(finite):
        abs_vals = torch.where(finite, abs_vals, torch.full_like(abs_vals, float("inf")))
    k = min(int(k), int(flat_v.numel()))
    top_vals, top_idx = torch.topk(abs_vals, k)
    sel_vals = flat_v[top_idx]
    sel_pts = X_train[top_idx]
    top_idx_cpu = top_idx.detach().cpu().tolist()
    top_vals_cpu = top_vals.detach().cpu().tolist()
    sel_vals_cpu = sel_vals.detach().cpu()
    sel_pts_cpu = sel_pts.detach().cpu()
    entries: List[Dict[str, Any]] = []
    for i, idx in enumerate(top_idx_cpu):
        entries.append(
            {
                "index": int(idx),
                "abs_value": float(top_vals_cpu[i]),
                "value": _scalar_to_json(sel_vals_cpu[i]),
                "x": [float(x) for x in sel_pts_cpu[i].tolist()],
            }
        )
    return entries


def build_vtrain_explosion_snapshot(
    spec: CanonicalSpec,
    X_train: torch.Tensor,
    V_train: torch.Tensor,
    A_train: Optional[torch.Tensor],
    *,
    layered_reference_enabled: bool,
    reference_subtracted_for_fit: bool,
    nan_to_num_applied: bool,
    clamp_applied: bool,
    seed: Optional[int] = None,
    gen: Optional[int] = None,
    program_idx: Optional[int] = None,
    topk: int = 8,
) -> Dict[str, Any]:
    total = int(V_train.numel())
    nonfinite = _nonfinite_count(V_train)
    frac_nonfinite = float(nonfinite / total) if total > 0 else 0.0
    charge_pos = _charge_positions(spec, X_train.device, X_train.dtype)
    payload = {
        "reason": "v_train_explosion",
        "seed": None if seed is None else int(seed),
        "gen": None if gen is None else int(gen),
        "program_idx": None if program_idx is None else int(program_idx),
        "flags": {
            "layered_reference_enabled": bool(layered_reference_enabled),
            "reference_subtracted_for_fit": bool(reference_subtracted_for_fit),
            "nan_to_num_applied": bool(nan_to_num_applied),
            "clamp_applied": bool(clamp_applied),
        },
        "stats": {
            "max_abs_V_train": _tensor_absmax(V_train),
            "frac_nonfinite_V_train": frac_nonfinite,
            "max_abs_A_train": _tensor_absmax(A_train),
            "min_distance_to_any_point_charge": _min_distance_to_charges(X_train, charge_pos),
        },
        "topk_abs_V_train": _topk_abs_entries(V_train, X_train, k=topk),
    }
    return payload


def write_vtrain_explosion_snapshot(out_dir: Path, payload: Dict[str, Any]) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_vtrain_snapshot.json"
    if out_path.exists():
        return False
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return True


__all__ = ["build_vtrain_explosion_snapshot", "write_vtrain_explosion_snapshot"]
