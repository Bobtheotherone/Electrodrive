from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from electrodrive.images.structural_features import structural_fingerprint
from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.basis import PointChargeBasis, annotate_group_info


_LIBRARY_CACHE: List[Dict[str, Any]] = []
_FLATTEN_CACHE: List[torch.Tensor] = []
_WEIGHT_VEC: torch.Tensor = torch.tensor([], dtype=torch.float64)
_NOVELTY_FAMILY_ORDER = [
    "axis_point",
    "three_layer_mirror",
    "three_layer_slab",
    "three_layer_tail",
    "three_layer_diffusion",
    "dcim_pole",
    "dcim_branch",
    "dcim_block",
    "layered_complex",
    "complex_depth_point",
]


def _load_library_configs_from_docs() -> List[Tuple[CanonicalSpec, ImageSystem]]:
    """Load representative (spec, system) pairs from Greens-function libraries."""
    pairs: List[Tuple[CanonicalSpec, ImageSystem]] = []
    library_paths = [
        Path("docs/research/electrostatics_greens_functions_library/library.json"),
        Path("docs/research/greens_functions_library/manifest.json"),
    ]
    for path in library_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        entries = []
        if isinstance(data, dict):
            if "entries" in data and isinstance(data["entries"], list):
                entries = data["entries"]
            elif "index" in data and isinstance(data["index"], dict):
                entries = data["index"].get("sections", [])

        for entry in entries:
            text = json.dumps(entry).lower()
            if all(k in text for k in ["layer", "slab"]) or "three_layer" in text:
                # Construct a minimal canonical three-layer spec.
                spec = CanonicalSpec.from_json(
                    {
                        "domain": {"bbox": [[-1, -1, -5], [1, 1, 5]]},
                        "dielectrics": [
                            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                            {"name": "slab", "epsilon": 4.0, "z_min": -0.5, "z_max": 0.0},
                            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.5},
                        ],
                        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
                        "BCs": "dielectric_interfaces",
                    }
                )
                elem_m = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.1])}, type_name="three_layer_images")
                annotate_group_info(elem_m, conductor_id=1, family_name="three_layer_mirror", motif_index=0)
                elem_s = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.3])}, type_name="three_layer_images")
                annotate_group_info(elem_s, conductor_id=1, family_name="three_layer_slab", motif_index=0)
                elem_t = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -1.0])}, type_name="three_layer_images")
                annotate_group_info(elem_t, conductor_id=2, family_name="three_layer_tail", motif_index=0)
                system = ImageSystem(
                    [elem_m, elem_s, elem_t],
                    torch.tensor([0.5, 0.4, 0.1], dtype=torch.float32),
                )
                pairs.append((spec, system))
                break
    return pairs


def _default_library_fingerprints() -> List[Dict[str, Any]]:
    """Small synthetic library to seed novelty scoring."""
    if _LIBRARY_CACHE:
        return _LIBRARY_CACHE

    real_pairs = _load_library_configs_from_docs()
    if real_pairs:
        for spec, system in real_pairs:
            _LIBRARY_CACHE.append(structural_fingerprint(system, spec))
    if _LIBRARY_CACHE:
        return _LIBRARY_CACHE

    # Build simple synthetic fingerprints for moderate + high-contrast slabs as fallback.
    from electrodrive.images.basis import PointChargeBasis, annotate_group_info

    def _make_system(z_list: List[float], fam: str) -> ImageSystem:
        elems = []
        weights = []
        for idx, z in enumerate(z_list):
            elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, z])}, type_name="three_layer_images")
            annotate_group_info(elem, conductor_id=1, family_name=fam, motif_index=idx)
            elems.append(elem)
            weights.append(1.0 / max(1, len(z_list)))
        return ImageSystem(elems, torch.tensor(weights, dtype=torch.float32))

    spec_data = {
        "domain": {"bbox": [[-1, -1, -2], [1, 1, 2]]},
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 1.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.5, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -2.0, "z_max": -0.5},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
    }
    spec = CanonicalSpec.from_json(spec_data)

    lib_fp = structural_fingerprint(
        _make_system([0.1, -0.2], "three_layer_slab"),
        spec,
    )
    high_fp = structural_fingerprint(
        _make_system([0.2, -0.8], "three_layer_tail"),
        spec,
    )
    _LIBRARY_CACHE.extend([lib_fp, high_fp])
    return _LIBRARY_CACHE


def _flatten_fingerprint(fp: Dict[str, Any]) -> torch.Tensor:
    families = fp.get("families", {})
    ladder = fp.get("ladder", {})
    vec: List[float] = []
    for fam in _NOVELTY_FAMILY_ORDER:
        fstats = families.get(fam, {})
        vec.extend(
            [
                float(fstats.get("count", 0.0)),
                float(fstats.get("weight_l1", 0.0)),
                float(fstats.get("weight_linf", 0.0)),
                float(fstats.get("z_norm_mean", 0.0)),
                float(fstats.get("z_norm_std", 0.0)),
                float(fstats.get("z_norm_min", 0.0)),
                float(fstats.get("z_norm_max", 0.0)),
            ]
        )
        lstats = ladder.get(fam, {"a": 0.0, "b": 0.0, "rms_resid": 0.0})
        vec.extend([float(lstats.get("a", 0.0)), float(lstats.get("b", 0.0)), float(lstats.get("rms_resid", 0.0))])
    sym = fp.get("symmetry", {})
    vec.append(float(sym.get("midplane_z_norm", 0.0)))
    vec.append(float(sym.get("asymmetry_metric", 0.0)))
    vec.append(float(fp.get("axis_weight_l1_fraction", 0.0)))
    vec.append(float(fp.get("nonaxis_weight_l1_fraction", 0.0)))
    discrete_ids = fp.get("discrete_ids", {})
    for key in ("interface_id", "schema_id"):
        stats = discrete_ids.get(key, {})
        vec.extend(
            [
                float(stats.get("count", 0.0)),
                float(stats.get("min", 0.0)),
                float(stats.get("max", 0.0)),
                float(stats.get("mean", 0.0)),
            ]
        )
    dcim_args = fp.get("dcim_args", {})
    vec.extend(
        [
            float(dcim_args.get("pole_count_sum", 0.0)),
            float(dcim_args.get("pole_count_max", 0.0)),
            float(dcim_args.get("branch_budget_sum", 0.0)),
            float(dcim_args.get("branch_budget_max", 0.0)),
        ]
    )
    return torch.tensor(vec, dtype=torch.float64)


def _weight_vector(dim: int) -> torch.Tensor:
    global _WEIGHT_VEC
    if _WEIGHT_VEC.numel() == dim:
        return _WEIGHT_VEC
    w = torch.ones(dim, dtype=torch.float64)
    # Heavier on weights and ladder spacing (positions 1,2 and ladder entries)
    for idx in range(dim):
        if idx % 10 in {1, 2}:  # weight_l1, weight_linf
            w[idx] = 2.5
        if idx % 10 in {7, 8}:  # ladder a, b
            w[idx] = 3.0
        if idx % 10 in {5, 6}:  # depth range
            w[idx] = 2.0
    tail = 4
    base = len(_NOVELTY_FAMILY_ORDER) * 10 + tail
    extra = dim - base
    w[-tail:] = torch.tensor([1.5, 1.5, 2.0, 2.0], dtype=torch.float64)
    if extra >= 12:
        start = dim - tail - 12
        w[start : start + 8] = 2.0
        w[start + 8 : start + 12] = 2.0
    _WEIGHT_VEC = w
    return w


def _prepare_library() -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    if _FLATTEN_CACHE:
        return _default_library_fingerprints(), torch.stack(_FLATTEN_CACHE)
    lib = _default_library_fingerprints()
    vecs = [_flatten_fingerprint(fp) for fp in lib]
    _FLATTEN_CACHE[:] = vecs
    return lib, torch.stack(vecs) if vecs else torch.zeros(0)


def novelty_score(fingerprint: Dict[str, Any]) -> float:
    lib, lib_vecs = _prepare_library()
    if lib_vecs.numel() == 0:
        return 1.0
    f_vec = _flatten_fingerprint(fingerprint)
    w = _weight_vector(f_vec.numel())
    diffs = lib_vecs - f_vec.unsqueeze(0)
    dists = torch.sqrt(torch.sum(w * diffs * diffs, dim=1))
    d_min = float(dists.min().item()) if dists.numel() else float("inf")

    # Calibrate using pairwise library distances.
    if lib_vecs.shape[0] >= 2:
        all_pairs = []
        for i in range(lib_vecs.shape[0]):
            for j in range(i + 1, lib_vecs.shape[0]):
                diff = lib_vecs[i] - lib_vecs[j]
                dist = torch.sqrt(torch.sum(w * diff * diff))
                all_pairs.append(float(dist.item()))
        if all_pairs:
            d_lo = float(torch.quantile(torch.tensor(all_pairs), 0.1).item())
            d_hi = float(torch.quantile(torch.tensor(all_pairs), 0.9).item())
        else:
            d_lo = d_hi = d_min
    else:
        d_lo = d_hi = d_min

    if d_hi <= d_lo:
        if d_min <= d_lo:
            return 0.0
        return 1.0
    if d_min <= d_lo:
        return 0.0
    if d_min >= d_hi:
        return 1.0
    return float((d_min - d_lo) / max(1e-12, (d_hi - d_lo)))


def compute_gate3_status(
    manifest: Dict[str, Any],
    novelty: float | None,
) -> Tuple[str, float | None]:
    gate1 = manifest.get("gate1_status", None)
    numeric = manifest.get("numeric_status", None)
    condition = manifest.get("condition_status", None)
    gate2 = manifest.get("gate2_status", None)

    if novelty is None:
        return "n/a", None
    if not (gate1 == "pass" and numeric == "ok" and condition != "ill_conditioned"):
        return "n/a", None
    if gate2 not in {"pass", "borderline"}:
        return "n/a", None
    if gate2 == "pass" and novelty >= 0.7:
        return "pass", novelty
    return "non_novel", novelty


def update_manifest_with_novelty(
    system: ImageSystem,
    spec: CanonicalSpec,
    manifest_path: Path,
) -> Tuple[float | None, str]:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    fp = structural_fingerprint(system, spec)
    novelty = novelty_score(fp) if manifest.get("gate2_status") in {"pass", "borderline"} else None
    gate3_status, novelty_val = compute_gate3_status(manifest, novelty)

    manifest["novelty_score"] = novelty_val if novelty is not None else None
    manifest["gate3_status"] = gate3_status
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["novelty_score"], gate3_status
