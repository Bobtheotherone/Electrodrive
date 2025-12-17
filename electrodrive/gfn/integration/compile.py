"""Compile GFlowNet programs into image basis elements."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock, StopProgram
from electrodrive.gfn.dsl.program import Program
from electrodrive.images.basis import ImageBasisElement, PointChargeBasis, annotate_group_info, compute_group_ids


def compile_program_to_basis(
    program: Program,
    spec: Any,
    device: torch.device,
) -> Tuple[List[ImageBasisElement], Sequence[Any], Mapping[str, Any]]:
    """Compile a program into basis elements compatible with the solver stack."""
    dtype = torch.float32
    elements: List[ImageBasisElement] = []
    warnings: List[str] = []
    family_counts: Dict[str, int] = {}

    spec_hash = _infer_spec_hash(spec)
    program_hash = program.hash(spec_hash)
    canonical_hex = program.canonical_bytes.hex()

    for node in program.nodes:
        if isinstance(node, StopProgram):
            continue
        if not isinstance(node, AddPrimitiveBlock):
            warnings.append(f"Skipped unsupported node type: {type(node).__name__}")
            continue

        family = str(node.family_name)
        conductor_id = int(node.conductor_id)
        motif_index = int(node.motif_id)

        element = _compile_primitive(
            family=family,
            conductor_id=conductor_id,
            motif_index=motif_index,
            spec=spec,
            device=device,
            dtype=dtype,
            warnings=warnings,
        )
        if element is None:
            continue

        info = getattr(element, "_group_info", {})
        family_name = str(info.get("family_name", element.type))
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
        elements.append(element)

    group_ids = compute_group_ids(elements, device=device, dtype=torch.long)
    meta = {
        "program_hash": program_hash,
        "program_canonical_hex": canonical_hex,
        "family_counts": family_counts,
        "warnings": warnings,
    }
    return elements, group_ids, meta


def _compile_primitive(
    *,
    family: str,
    conductor_id: int,
    motif_index: int,
    spec: Any,
    device: torch.device,
    dtype: torch.dtype,
    warnings: List[str],
) -> ImageBasisElement | None:
    type_name = _resolve_basis_type_name(family)
    pool = _candidate_pool(family, spec, device=device, dtype=dtype, conductor_id=conductor_id)
    if not pool:
        warnings.append(f"No candidates for family '{family}'")
        return None
    idx = motif_index % len(pool)
    pos, family_name = pool[idx]
    elem = PointChargeBasis({"position": pos}, type_name=type_name)
    annotate_group_info(
        elem,
        conductor_id=conductor_id,
        family_name=family_name,
        motif_index=motif_index,
    )
    return elem


def _resolve_basis_type_name(family: str) -> str:
    if family == "baseline":
        return "point"
    if family == "axis_point":
        return "axis_point"
    if family.startswith("three_layer"):
        return "three_layer_images"
    return "point"


def _candidate_pool(
    family: str,
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    conductor_id: int,
) -> List[Tuple[torch.Tensor, str]]:
    if family in {"point", "baseline"}:
        return [(pos, "point") for pos in _point_candidates(spec, device=device, dtype=dtype, conductor_id=conductor_id)]
    if family == "learned_point":
        return [(pos, "learned_point") for pos in _point_candidates(spec, device=device, dtype=dtype, conductor_id=conductor_id)]
    if family == "axis_point":
        return [(pos, "axis_point") for pos in _axis_point_candidates(spec, device=device, dtype=dtype, conductor_id=conductor_id)]
    if family in {"three_layer_images", "three_layer_mirror", "three_layer_slab", "three_layer_tail"}:
        candidates = _three_layer_candidates(spec, device=device, dtype=dtype)
        if family == "three_layer_images":
            return candidates
        return [(pos, family) for pos, _ in candidates]
    return []


def _point_candidates(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    conductor_id: int,
) -> List[torch.Tensor]:
    charges = _extract_charge_positions(spec, device=device, dtype=dtype)
    if not charges:
        return []
    plane = _select_conductor(spec, conductor_id, target_type="plane")
    if plane is None:
        plane = _select_conductor(spec, None, target_type="plane")
    z_plane = float(plane.get("z", 0.0)) if plane else None

    positions: List[torch.Tensor] = []
    for pos in charges:
        positions.append(pos)
        if z_plane is None:
            continue
        x0, y0, z0 = float(pos[0].item()), float(pos[1].item()), float(pos[2].item())
        z_img = 2.0 * z_plane - z0
        img_pos = torch.tensor([x0, y0, z_img], device=device, dtype=dtype)
        positions.append(img_pos)
    return positions


def _axis_point_candidates(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    conductor_id: int,
) -> List[torch.Tensor]:
    spheres = _select_conductors(spec, target_type="sphere")
    if not spheres:
        return []
    sph = spheres[conductor_id] if 0 <= conductor_id < len(spheres) else spheres[0]
    center = torch.tensor(sph.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype).view(3)
    radius = float(sph.get("radius", 1.0))

    charges = _extract_charge_positions(spec, device=device, dtype=dtype)
    if charges:
        dists = [torch.linalg.norm(p - center).item() for p in charges]
        idx = int(min(range(len(dists)), key=lambda i: dists[i]))
        axis_dir = charges[idx] - center
    else:
        axis_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    if float(torch.linalg.norm(axis_dir).item()) < 1e-9:
        axis_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    axis_dir = axis_dir / (torch.linalg.norm(axis_dir) + 1e-12)

    distances = [0.3 * radius, 0.6 * radius, 0.9 * radius, 1.1 * radius]
    return [center + dist * axis_dir for dist in distances]


def _three_layer_candidates(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, str]]:
    if getattr(spec, "BCs", "") != "dielectric_interfaces":
        return []
    layers = getattr(spec, "dielectrics", None) or []
    if len(layers) != 3:
        return []

    triples: List[Tuple[float, float]] = []
    for layer in layers:
        try:
            z_min = float(layer["z_min"])
            z_max = float(layer["z_max"])
        except Exception:
            return []
        triples.append((z_min, z_max))
    triples.sort(key=lambda t: (t[0] + t[1]) * 0.5)
    bottom, middle, top = triples
    bottom_z = bottom[1]
    top_z = top[0]
    h = max(top_z - bottom_z, 1e-6)

    candidates: List[Tuple[torch.Tensor, str]] = []
    charges = _extract_charge_positions(spec, device=device, dtype=dtype)
    if not charges:
        return []
    for pos in charges:
        x0, y0, z0 = float(pos[0].item()), float(pos[1].item()), float(pos[2].item())
        z_candidates = [
            2.0 * top_z - z0,
            2.0 * top_z - z0 - 2.0 * h,
            top_z - 0.25 * h,
            top_z - 0.75 * h,
            bottom_z - 0.5 * h,
            bottom_z - 1.5 * h,
            bottom_z - 3.0 * h,
        ]
        for idx_img, z_img in enumerate(z_candidates):
            if idx_img < 2:
                family = "three_layer_mirror"
            elif idx_img < 4:
                family = "three_layer_slab"
            else:
                family = "three_layer_tail"
            candidates.append((torch.tensor([x0, y0, z_img], device=device, dtype=dtype), family))
    return candidates


def _extract_charge_positions(
    spec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    charges = getattr(spec, "charges", None) or []
    positions: List[torch.Tensor] = []
    for ch in charges:
        if ch.get("type") != "point":
            continue
        pos_raw = ch.get("pos", None)
        if pos_raw is None:
            continue
        positions.append(torch.tensor(pos_raw, device=device, dtype=dtype).view(3))
    return positions


def _select_conductor(spec: Any, conductor_id: int | None, target_type: str) -> Dict[str, Any] | None:
    conductors = getattr(spec, "conductors", None) or []
    if conductor_id is not None and 0 <= conductor_id < len(conductors):
        if conductors[conductor_id].get("type") == target_type:
            return conductors[conductor_id]
    for cond in conductors:
        if cond.get("type") == target_type:
            return cond
    return None


def _select_conductors(spec: Any, target_type: str) -> List[Dict[str, Any]]:
    conductors = getattr(spec, "conductors", None) or []
    return [cond for cond in conductors if cond.get("type") == target_type]


def _infer_spec_hash(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    for attr in ("spec_hash", "hash", "id"):
        if hasattr(spec, attr):
            return str(getattr(spec, attr))
    return str(spec)


__all__ = ["compile_program_to_basis"]
