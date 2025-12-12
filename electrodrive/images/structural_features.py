from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec


_FAMILY_ORDER: List[str] = [
    "axis_point",
    "three_layer_mirror",
    "three_layer_slab",
    "three_layer_tail",
    "three_layer_diffusion",
]
_LADDER_FAMS = {"three_layer_slab", "three_layer_tail"}


def _family_name(elem: Any) -> str:
    info = getattr(elem, "_group_info", None)
    if isinstance(info, dict) and info.get("family_name"):
        return str(info["family_name"])
    return getattr(elem, "type", "unknown")


def _position(elem: Any) -> Tuple[float, float, float] | None:
    pos = getattr(elem, "params", {}).get("position", None)
    if isinstance(pos, torch.Tensor):
        p = pos.detach().cpu().view(-1)
        if p.numel() >= 3:
            return float(p[0]), float(p[1]), float(p[2])
    return None


def _norm_depth(z: float, z_top: float, h: float) -> float:
    if h <= 0:
        return 0.0
    return (z - z_top) / h


def _family_stats(z_norms: List[float], weights: List[float]) -> Dict[str, float]:
    if not z_norms:
        return {
            "z_norm_mean": 0.0,
            "z_norm_std": 0.0,
            "z_norm_min": 0.0,
            "z_norm_max": 0.0,
        }
    z = torch.tensor(z_norms, dtype=torch.float64)
    return {
        "z_norm_mean": float(z.mean().item()),
        "z_norm_std": float(z.std(unbiased=False).item()),
        "z_norm_min": float(z.min().item()),
        "z_norm_max": float(z.max().item()),
    }


def _ladder_fit(z_norms: List[float]) -> Dict[str, float]:
    if len(z_norms) < 2:
        mean_val = float(z_norms[0]) if z_norms else 0.0
        return {"a": 0.0, "b": mean_val, "rms_resid": 0.0}
    z_sorted = sorted(z_norms)
    k = torch.arange(len(z_sorted), dtype=torch.float64)
    z = torch.tensor(z_sorted, dtype=torch.float64)
    A = torch.stack([k, torch.ones_like(k)], dim=1)
    lstsq_out = torch.linalg.lstsq(A, z)
    sol = lstsq_out.solution
    a, b = float(sol[0].item()), float(sol[1].item())
    pred = a * k + b
    rms = float(torch.sqrt(torch.mean((z - pred) ** 2)).item())
    return {"a": a, "b": b, "rms_resid": rms}


def structural_fingerprint(system: ImageSystem, spec: CanonicalSpec) -> Dict[str, Any]:
    """
    Extract a fixed-schema structural fingerprint for a discovered image system.
    """
    families: Dict[str, Dict[str, Any]] = {
        fam: {
            "count": 0,
            "weight_l1": 0.0,
            "weight_linf": 0.0,
            "z_norm_mean": 0.0,
            "z_norm_std": 0.0,
            "z_norm_min": 0.0,
            "z_norm_max": 0.0,
        }
        for fam in _FAMILY_ORDER
    }
    ladder: Dict[str, Dict[str, float]] = {
        fam: {"a": 0.0, "b": 0.0, "rms_resid": 0.0} for fam in _LADDER_FAMS
    }

    # Slab geometry normalization.
    z_top = 0.0
    z_bottom = 0.0
    h = 1.0
    if getattr(spec, "BCs", "") == "dielectric_interfaces":
        layers = getattr(spec, "dielectrics", None) or []
        if len(layers) == 3:
            try:
                z_vals: List[float] = []
                for layer in layers:
                    z_vals.extend([float(layer["z_min"]), float(layer["z_max"])])
                z_top = max(z_vals)
                z_bottom = min(z_vals)
                h = max(1e-6, z_top - z_bottom)
            except Exception:
                pass

    axis_weight = 0.0
    total_weight = 0.0

    per_family_z: Dict[str, List[float]] = {fam: [] for fam in _FAMILY_ORDER}
    per_family_w: Dict[str, List[float]] = {fam: [] for fam in _FAMILY_ORDER}

    for elem, w in zip(system.elements, system.weights.detach().cpu()):
        fam = _family_name(elem)
        if fam not in families:
            continue
        pos = _position(elem)
        z_norm = 0.0
        if pos is not None:
            z_norm = _norm_depth(pos[2], z_top, h)
        w_abs = float(abs(float(w)))
        total_weight += w_abs
        if fam == "axis_point":
            axis_weight += w_abs
        per_family_z[fam].append(z_norm)
        per_family_w[fam].append(w_abs)
        fam_stats = families[fam]
        fam_stats["count"] += 1
        fam_stats["weight_l1"] += w_abs
        fam_stats["weight_linf"] = max(fam_stats["weight_linf"], w_abs)

    for fam in _FAMILY_ORDER:
        stats = _family_stats(per_family_z[fam], per_family_w[fam])
        families[fam].update(stats)

    for fam in _LADDER_FAMS:
        ladder[fam] = _ladder_fit(per_family_z.get(fam, []))

    # Symmetry proxy: mid-plane and asymmetry around it.
    midplane_z = (z_top + z_bottom) * 0.5
    midplane_norm = _norm_depth(midplane_z, z_top, h)
    all_z = []
    for fam in _FAMILY_ORDER:
        all_z.extend(per_family_z.get(fam, []))
    asym = 0.0
    if all_z:
        zs = sorted(all_z)
        pairs = zip(zs, reversed(zs))
        diffs = []
        for a, b in pairs:
            diffs.append(abs(a + b))
        if diffs:
            asym = float(sum(diffs) / len(diffs))

    axis_frac = (axis_weight / total_weight) if total_weight > 0 else 0.0
    nonaxis_frac = 1.0 - axis_frac if total_weight > 0 else 0.0

    return {
        "families": families,
        "ladder": ladder,
        "symmetry": {
            "midplane_z_norm": midplane_norm,
            "asymmetry_metric": asym,
        },
        "axis_weight_l1_fraction": axis_frac,
        "nonaxis_weight_l1_fraction": nonaxis_frac,
    }
