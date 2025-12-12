from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from electrodrive.images.io import load_image_system
from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec


def _load_spec(path: Path) -> CanonicalSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CanonicalSpec.from_json(data)


def _collect_domain_extent(spec: CanonicalSpec) -> float:
    """Estimate a reference z-extent from charges, dielectric bounds, and domain bbox."""
    z_vals: List[float] = []
    for ch in spec.charges or []:
        try:
            z_vals.append(abs(float(ch["pos"][2])))
        except Exception:
            continue
    for layer in spec.dielectrics or []:
        for key in ("z_min", "z_max"):
            if key in layer:
                try:
                    z_vals.append(abs(float(layer[key])))
                except Exception:
                    continue
    domain = spec.domain
    if isinstance(domain, dict):
        bbox = domain.get("bbox", None)
        if bbox and len(bbox) == 2:
            try:
                z_vals.append(abs(float(bbox[0][2])))
                z_vals.append(abs(float(bbox[1][2])))
            except Exception:
                pass
    max_z = max(z_vals) if z_vals else 1.0
    return max(1.0, max_z)


def _find_repo_root(start: Path) -> Path:
    """Walk upward to locate the repo root (identified by .git)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _resolve_spec_path(spec_arg: str | None, manifest: Dict[str, Any], run_path: Path) -> Path:
    """Resolve the spec path robustly using provided arg or manifest spec_path."""
    candidates = []
    if spec_arg:
        candidates.append(Path(spec_arg))
    manifest_spec = manifest.get("spec_path")
    if manifest_spec:
        candidates.append(Path(manifest_spec))

    repo_root = _find_repo_root(Path(__file__).resolve())
    for cand in candidates:
        if cand.is_absolute() and cand.exists():
            return cand
        if cand.exists():
            return cand
        alt = repo_root / cand
        if alt.exists():
            return alt
    if not candidates:
        raise FileNotFoundError("No spec path provided and manifest missing spec_path.")
    raise FileNotFoundError(f"Could not resolve spec path from candidates: {candidates}")


def _position_from_elem(elem: Any) -> Tuple[float, float, float] | None:
    pos = None
    if hasattr(elem, "params"):
        p = elem.params
        if isinstance(p, dict) and "position" in p:
            try:
                v = p["position"]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().view(-1).tolist()
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    pos = (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                pos = None
    return pos


def _family_name(elem: Any) -> str:
    info = getattr(elem, "_group_info", None)
    if isinstance(info, dict) and info.get("family_name"):
        try:
            return str(info["family_name"])
        except Exception:
            pass
    return getattr(elem, "type", "unknown")


def _compute_degeneracies(
    elements: List[Any],
    weights: torch.Tensor,
    z_threshold: float,
) -> Dict[str, int]:
    tol_pos = 1e-6
    tol_w = 0.05

    positions: List[Tuple[float, float, float]] = []
    for elem in elements:
        pos = _position_from_elem(elem)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append((float("nan"), float("nan"), float("nan")))

    n_dup = 0
    for i in range(len(elements)):
        pi = positions[i]
        if any(map(lambda x: x != x, pi)):  # nan check
            continue
        wi = float(weights[i])
        for j in range(i + 1, len(elements)):
            pj = positions[j]
            if any(map(lambda x: x != x, pj)):
                continue
            dx = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2
            if dx ** 0.5 < tol_pos:
                wj = float(weights[j])
                denom = max(abs(wi), abs(wj), 1e-12)
                if abs(wi - wj) <= tol_w * denom:
                    n_dup += 1

    n_far = 0
    for pos in positions:
        if any(map(lambda x: x != x, pos)):
            continue
        if abs(pos[2]) > 2.0 * z_threshold:
            n_far += 1

    return {
        "duplicate_physical_charges": n_dup,
        "far_tails_over_domain": n_far,
    }


def compute_structural_summary(
    spec: CanonicalSpec,
    system: ImageSystem,
    *,
    numeric_status: str | None = "ok",
    condition_status: str | None = None,
) -> Dict[str, Any]:
    """Compute per-family structural metrics and Gate 2 score."""
    elements = system.elements
    weights = system.weights.detach().cpu()

    summary: Dict[str, Any] = {
        "n_images": len(elements),
        "families": {},
        "degeneracies": {},
        "structure_score": None,
        "gate2_status": "n/a",
    }

    numeric_ok = (numeric_status or "").lower() == "ok"
    condition_ok = (condition_status or "").lower() != "ill_conditioned"
    if not numeric_ok or not condition_ok:
        summary["note"] = "Numeric or conditioning failed; structural scoring is diagnostic only."
        return summary

    family_metrics: Dict[str, Dict[str, Any]] = {}
    weight_total = 0.0
    positions_by_family: Dict[str, List[float]] = {}
    for elem, w in zip(elements, weights):
        fam = _family_name(elem)
        fam_metrics = family_metrics.setdefault(
            fam,
            {"count": 0, "weight_l1": 0.0, "weight_linf": 0.0, "z_min": None, "z_max": None},
        )
        w_abs = float(abs(float(w)))
        fam_metrics["count"] += 1
        fam_metrics["weight_l1"] += w_abs
        fam_metrics["weight_linf"] = max(fam_metrics["weight_linf"], w_abs)
        weight_total += w_abs

        pos = _position_from_elem(elem)
        if pos is not None:
            z = float(pos[2])
            fam_metrics["z_min"] = z if fam_metrics["z_min"] is None else min(fam_metrics["z_min"], z)
            fam_metrics["z_max"] = z if fam_metrics["z_max"] is None else max(fam_metrics["z_max"], z)
            positions_by_family.setdefault(fam, []).append(z)

    summary["families"] = family_metrics

    z_ref = _collect_domain_extent(spec)
    degeneracies = _compute_degeneracies(elements, weights, z_ref)
    summary["degeneracies"] = degeneracies

    n_dup = float(degeneracies.get("duplicate_physical_charges", 0))
    n_far = float(degeneracies.get("far_tails_over_domain", 0))
    w_total = weight_total
    w_axis = family_metrics.get("axis_point", {}).get("weight_l1", 0.0)
    w_nonaxis = max(0.0, w_total - w_axis)

    slab_fams = ("three_layer_slab", "three_layer_mirror", "three_layer_tail")
    slab_weight = sum(family_metrics.get(f, {}).get("weight_l1", 0.0) for f in slab_fams)
    frac_slab = (slab_weight / w_total) if w_total > 0 else 0.0
    F_slab_active = 1 if frac_slab >= 0.05 else 0

    S = 1.0
    S -= min(0.2, 0.05 * n_dup)
    S -= min(0.2, 0.02 * n_far)
    if w_total > 0 and (w_nonaxis / w_total) < 0.1:
        S -= 0.4
    if F_slab_active == 0:
        S -= 0.3
    S = max(0.0, min(1.0, S))

    summary["structure_score"] = S

    if S >= 0.7 and F_slab_active == 1:
        gate2 = "pass"
    elif 0.4 <= S < 0.7:
        gate2 = "borderline"
    else:
        gate2 = "fail"

    summary["gate2_status"] = gate2
    return summary


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _update_manifest(manifest_path: Path, structure_score: Any, gate2_status: str) -> None:
    manifest = _load_manifest(manifest_path)
    manifest["structure_score"] = structure_score
    manifest["gate2_status"] = gate2_status
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 2 structural diagnostics for discovered image systems.")
    ap.add_argument("--run", required=True, help="Path to discovered_system.json")
    ap.add_argument(
        "--spec",
        required=False,
        default=None,
        help="Path to original spec JSON (defaults to manifest spec_path when omitted)",
    )
    ap.add_argument("--out", required=True, help="Where to write gate2 summary JSON")
    ap.add_argument("--device", default="cpu", help="Device for tensor reconstruction (default: cpu)")
    ap.add_argument("--dtype", default="float32", help="Torch dtype for reconstruction (default: float32)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    run_path = Path(args.run)
    out_path = Path(args.out)

    manifest_path = run_path.parent / "discovery_manifest.json"
    manifest = _load_manifest(manifest_path)
    spec_path = _resolve_spec_path(args.spec, manifest, run_path)
    spec = _load_spec(spec_path)
    system = load_image_system(run_path, device=args.device, dtype=dtype)

    numeric_status = manifest.get("numeric_status", "ok")
    condition_status = manifest.get("condition_status", None)

    summary = compute_structural_summary(
        spec,
        system,
        numeric_status=numeric_status,
        condition_status=condition_status,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Update manifest with gate2 fields even when diagnostic-only
    _update_manifest(manifest_path, summary.get("structure_score"), summary.get("gate2_status"))


if __name__ == "__main__":
    main()
