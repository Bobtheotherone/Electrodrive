"""Compile GFlowNet programs into image basis elements."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import math

import torch

from electrodrive.flows.device_guard import ensure_cuda
from electrodrive.flows.schemas import REGISTRY, SCHEMA_REAL_POINT
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl.nodes import (
    AddBranchCutBlock,
    AddPoleBlock,
    AddPrimitiveBlock,
    ConjugatePair,
    StopProgram,
)
from electrodrive.gfn.dsl.program import Program
from electrodrive.images.basis import (
    DCIMBranchCutImageBasis,
    DCIMPoleImageBasis,
    ImageBasisElement,
    PointChargeBasis,
    annotate_group_info,
    compute_group_ids,
)


def compile_program_to_basis(
    program: Program,
    spec: Any,
    device: torch.device,
    *,
    param_payload: ParamPayload | None = None,
    strict: bool | None = None,
) -> Tuple[List[ImageBasisElement], Sequence[Any], Mapping[str, Any]]:
    """Compile a program into basis elements compatible with the solver stack."""
    strict_payload = _strict_param_payload() if strict is None else bool(strict)
    if strict_payload and param_payload is None:
        raise RuntimeError("ParamPayload required when strict param payload is enabled.")

    dtype = torch.float32
    elements: List[ImageBasisElement] = []
    warnings: List[str] = []
    family_counts: Dict[str, int] = {}

    spec_hash = _infer_spec_hash(spec)
    program_hash = program.hash(spec_hash)
    canonical_hex = program.canonical_bytes.hex()

    payload = None
    node_to_token: List[int] = []
    if param_payload is not None:
        payload = _coerce_payload(param_payload)
        device = ensure_cuda(payload.device)
        if not payload.u_latent.is_cuda:
            raise ValueError("ParamPayload.u_latent must be on CUDA.")
        if payload.device.type != "cuda":
            raise ValueError(f"ParamPayload.device must be CUDA, got {payload.device}.")
        dtype = payload.dtype
        node_to_token = _normalize_node_to_token(payload.node_to_token, len(program.nodes))

    layered_ctx = _layered_context(spec, device=device, dtype=dtype)
    is_layered = layered_ctx is not None

    node_to_element: Dict[int, int] = {}
    for node_idx, node in enumerate(program.nodes):
        if isinstance(node, StopProgram):
            continue
        if isinstance(node, ConjugatePair):
            if is_layered:
                # Layered DCIM uses folded complex pairs; skip explicit conjugate nodes.
                continue
            element = _compile_conjugate_pair(node, elements, node_to_element, device, dtype, warnings)
            if element is None:
                continue
        elif payload is not None:
            element = _compile_param_node(
                node,
                node_idx,
                payload,
                node_to_token,
                spec,
                device,
                dtype,
                layered_ctx,
                warnings,
            )
            if element is None:
                continue
        else:
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
                layered_ctx=layered_ctx,
                warnings=warnings,
            )
            if element is None:
                continue

        info = getattr(element, "_group_info", {})
        family_name = str(info.get("family_name", element.type))
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
        elements.append(element)
        node_to_element[node_idx] = len(elements) - 1

    group_ids = compute_group_ids(elements, device=device, dtype=torch.long)
    meta = {
        "program_hash": program_hash,
        "program_canonical_hex": canonical_hex,
        "family_counts": family_counts,
        "warnings": warnings,
    }
    return elements, group_ids, meta


def _strict_param_payload() -> bool:
    raw = os.getenv("EDE_STRICT_PARAM_PAYLOAD", "").strip().lower()
    return raw not in ("", "0", "false", "no")


def _coerce_payload(payload: ParamPayload) -> ParamPayload:
    needs_slice = False
    if payload.u_latent.dim() >= 3:
        if payload.u_latent.shape[0] != 1:
            raise ValueError("ParamPayload is batched; call ParamPayload.for_program() first.")
        needs_slice = True
    if payload.node_mask.dim() >= 2 and payload.node_mask.shape[0] == 1:
        needs_slice = True
    if payload.schema_ids.dim() >= 2 and payload.schema_ids.shape[0] == 1:
        needs_slice = True
    if isinstance(payload.node_to_token, torch.Tensor) and payload.node_to_token.dim() >= 2:
        if payload.node_to_token.shape[0] == 1:
            needs_slice = True
    if needs_slice:
        return payload.for_program(0)
    return payload


def _normalize_node_to_token(node_to_token: Sequence[Sequence[int]] | torch.Tensor, count: int) -> List[int]:
    mapping: List[int]
    if isinstance(node_to_token, torch.Tensor):
        if node_to_token.is_cuda:
            raise ValueError("node_to_token must be CPU metadata to avoid GPU sync.")
        if node_to_token.dim() > 1:
            if node_to_token.shape[0] != 1:
                raise ValueError("ParamPayload.node_to_token is batched; call ParamPayload.for_program().")
            node_to_token = node_to_token[0]
        mapping = [int(v) for v in node_to_token.detach().tolist()]
    else:
        if node_to_token and isinstance(node_to_token[0], (list, tuple)):
            if len(node_to_token) != 1:
                raise ValueError("ParamPayload.node_to_token is batched; call ParamPayload.for_program().")
            node_to_token = node_to_token[0]
        mapping = [int(v) for v in node_to_token]
    if len(mapping) < count:
        mapping.extend([-1] * (count - len(mapping)))
    return mapping


def _compile_conjugate_pair(
    node: ConjugatePair,
    elements: Sequence[ImageBasisElement],
    node_to_element: Mapping[int, int],
    device: torch.device,
    dtype: torch.dtype,
    warnings: List[str],
) -> ImageBasisElement | None:
    try:
        ref = int(node.block_ref)
    except Exception:
        warnings.append(f"Invalid conjugate_pair block_ref: {node.block_ref}")
        return None
    elem_idx = node_to_element.get(ref)
    if elem_idx is None or elem_idx < 0 or elem_idx >= len(elements):
        warnings.append(f"ConjugatePair missing reference element for block_ref={ref}")
        return None
    base = elements[elem_idx]
    params = {k: v.clone() for k, v in base.params.items()}
    z_imag = params.get("z_imag")
    if z_imag is None:
        z_imag = torch.zeros((), device=device, dtype=dtype)
    else:
        z_imag = z_imag.to(device=device, dtype=dtype)
    params["z_imag"] = -z_imag
    if isinstance(base, PointChargeBasis) and base.__class__ is PointChargeBasis:
        elem = PointChargeBasis(params, type_name=base.type)
    else:
        try:
            elem = base.__class__(params)
        except Exception:
            warnings.append(f"ConjugatePair unsupported basis type: {type(base).__name__}")
            return None
    info = getattr(base, "_group_info", None)
    if isinstance(info, dict):
        setattr(elem, "_group_info", dict(info))
    return elem


def _compile_param_node(
    node: object,
    node_idx: int,
    payload: ParamPayload,
    node_to_token: Sequence[int],
    spec: Any,
    device: torch.device,
    dtype: torch.dtype,
    layered_ctx: Dict[str, Any] | None,
    warnings: List[str],
) -> ImageBasisElement | None:
    if not isinstance(node, (AddPrimitiveBlock, AddPoleBlock, AddBranchCutBlock)):
        warnings.append(f"Skipped unsupported node type: {type(node).__name__}")
        return None

    token_idx = node_to_token[node_idx] if node_idx < len(node_to_token) else -1
    if token_idx < 0:
        warnings.append(f"Missing token slot for node index {node_idx}")
        return None

    if payload.node_mask.numel() > 0:
        try:
            if not bool(payload.node_mask[token_idx].item()):
                warnings.append(f"Token slot masked for node index {node_idx}")
                return None
        except Exception:
            warnings.append(f"Invalid node_mask for token index {token_idx}")
            return None

    u_latent = payload.u_latent
    if u_latent.dim() == 2:
        u = u_latent[token_idx]
    elif u_latent.dim() == 1:
        u = u_latent
    else:
        warnings.append("ParamPayload.u_latent has unsupported rank.")
        return None

    schema_id = _resolve_schema_id(payload, node, token_idx, warnings)
    schema = REGISTRY.get_by_id(schema_id)
    if schema is None:
        warnings.append(f"Unknown schema_id {schema_id}; falling back to real_point.")
        schema = REGISTRY.get_by_id(SCHEMA_REAL_POINT)
    if schema is None:
        warnings.append("Schema registry missing default real_point.")
        return None

    node_ctx = _node_context(node)
    params = schema.transform(u, spec, node_ctx=node_ctx)
    params = _apply_layered_param_overrides(
        node,
        node_idx,
        params,
        layered_ctx,
        device=device,
        dtype=dtype,
    )
    elem = _element_from_params(node, params, device, dtype, warnings)
    return elem


def _resolve_schema_id(
    payload: ParamPayload,
    node: object,
    token_idx: int,
    warnings: List[str],
) -> int:
    schema_id = 0
    if payload.schema_ids.numel() > 0:
        try:
            schema_id = int(payload.schema_ids[token_idx].item())
        except Exception:
            schema_id = 0

    node_schema_id = getattr(node, "schema_id", None)
    if node_schema_id is not None:
        try:
            node_schema_id = int(node_schema_id)
        except Exception:
            node_schema_id = None

    if schema_id <= 0 and node_schema_id:
        schema_id = node_schema_id
    elif schema_id > 0 and node_schema_id and node_schema_id != schema_id:
        warnings.append(f"Schema id mismatch: payload={schema_id}, node={node_schema_id}")

    if schema_id <= 0:
        schema_name = getattr(node, "schema_name", None)
        if schema_name:
            schema = REGISTRY.get_by_name(str(schema_name))
            if schema is not None:
                schema_id = int(schema.schema_id)

    if schema_id <= 0:
        schema_id = SCHEMA_REAL_POINT
    return schema_id


def _node_context(node: object) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for key in ("conductor_id", "interface_id", "family_name", "motif_type", "schema_id", "schema_name"):
        if hasattr(node, key):
            value = getattr(node, key)
            if value is not None:
                ctx[key] = value
    return ctx


def _element_from_params(
    node: object,
    params: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    warnings: List[str],
) -> ImageBasisElement | None:
    family = getattr(node, "family_name", None)
    schema_name = getattr(node, "schema_name", None)
    family_name = None
    if family is not None:
        family_name = str(family)
    elif schema_name is not None:
        family_name = str(schema_name)
    type_name = _resolve_basis_type_name(family_name) if family_name is not None else "point"
    position = params.get("position")
    if position is None:
        warnings.append(f"Param node missing position for {type(node).__name__}")
        return None

    pos = position.to(device=device, dtype=dtype)
    params_out = dict(params)
    params_out["position"] = pos
    if "z_imag" not in params_out:
        params_out["z_imag"] = torch.zeros((), device=device, dtype=dtype)
    if isinstance(node, AddPoleBlock):
        elem = DCIMPoleImageBasis(params_out)
    elif isinstance(node, AddBranchCutBlock):
        elem = DCIMBranchCutImageBasis(params_out)
    else:
        elem = PointChargeBasis(params_out, type_name=type_name)

    conductor_id = _coerce_int(getattr(node, "conductor_id", 0), 0)
    motif_index = _coerce_int(getattr(node, "motif_id", 0), 0)
    annotate_group_info(
        elem,
        conductor_id=conductor_id,
        family_name=family_name,
        motif_index=motif_index,
    )
    return elem


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _layered_context(spec: Any, *, device: torch.device, dtype: torch.dtype) -> Dict[str, Any] | None:
    if getattr(spec, "BCs", "") != "dielectric_interfaces":
        return None
    source_vals = _layered_source_values(spec)
    source_pos = (
        torch.tensor(source_vals, device=device, dtype=dtype).view(3) if source_vals is not None else None
    )
    slab = _layered_slab_info(spec)
    return {"source_vals": source_vals, "source_pos": source_pos, "slab": slab}


def _layered_source_values(spec: Any) -> Tuple[float, float, float] | None:
    charges = getattr(spec, "charges", None) or []
    for ch in charges:
        if ch.get("type") != "point":
            continue
        pos = ch.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            continue
        try:
            return float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            continue
    return None


def _layered_slab_info(spec: Any) -> Dict[str, float] | None:
    layers = getattr(spec, "dielectrics", None) or []
    best = None
    best_thickness = None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        try:
            z_min = float(layer.get("z_min"))
            z_max = float(layer.get("z_max"))
        except Exception:
            continue
        thickness = abs(z_max - z_min)
        if thickness <= 0.0:
            continue
        eps_raw = layer.get("epsilon", layer.get("eps", 1.0))
        try:
            eps_val = float(eps_raw)
        except Exception:
            eps_val = 1.0
        info = {
            "z_min": min(z_min, z_max),
            "z_max": max(z_min, z_max),
            "eps": eps_val,
            "h": thickness,
        }
        if layer.get("name") == "slab":
            return info
        if best_thickness is None or thickness < best_thickness:
            best_thickness = thickness
            best = info
    return best


def _eps_scale(eps: float | None) -> float:
    if eps is None:
        return 1.0
    try:
        eps_val = float(eps)
    except Exception:
        return 1.0
    eps_val = max(eps_val, 0.0)
    scale = math.log1p(eps_val)
    scale = min(max(scale, 1.0), 5.0) / 5.0
    return scale


def _apply_layered_xy_override(
    pos: torch.Tensor,
    layered_ctx: Dict[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if layered_ctx is None:
        return pos.to(device=device, dtype=dtype)
    source_pos = layered_ctx.get("source_pos")
    if source_pos is None:
        return pos.to(device=device, dtype=dtype)
    pos = pos.to(device=device, dtype=dtype).view(3)
    pos = pos.clone()
    pos[:2] = source_pos[:2]
    return pos


def _apply_layered_param_overrides(
    node: object,
    node_idx: int,
    params: Mapping[str, torch.Tensor],
    layered_ctx: Dict[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    if layered_ctx is None:
        return dict(params)

    params_out: Dict[str, torch.Tensor] = dict(params)
    pos = params_out.get("position")
    if pos is not None:
        pos = _apply_layered_xy_override(pos, layered_ctx, device=device, dtype=dtype)
        params_out["position"] = pos

    if not isinstance(node, (AddPoleBlock, AddBranchCutBlock)):
        return params_out

    slab = layered_ctx.get("slab")
    source_vals = layered_ctx.get("source_vals")
    if slab is None or source_vals is None:
        return params_out

    h = max(float(slab.get("h", 0.0)), 1e-6)
    z0 = float(source_vals[2])
    k = _coerce_int(getattr(node, "motif_id", node_idx), node_idx)
    scale = _eps_scale(slab.get("eps"))

    if isinstance(node, AddPoleBlock):
        alpha = (0.15, 0.35, 0.75)[k % 3]
        z_real_default = -(z0 + 2.0 * k * h)
        z_imag_default = alpha * h * scale
    else:
        beta = (0.6, 1.2, 2.0)[k % 3]
        z_real_default = -(z0 + (2.0 * k + 1.0) * h)
        z_imag_default = beta * h * scale

    if pos is None:
        pos = torch.tensor([source_vals[0], source_vals[1], z_real_default], device=device, dtype=dtype)
    else:
        pos = pos.to(device=device, dtype=dtype).view(3)

    z_imag = params_out.get("z_imag")
    if z_imag is None:
        z_imag = torch.zeros((), device=device, dtype=dtype)
    z_imag = z_imag.to(device=device, dtype=dtype).view(())

    min_imag = max(1e-3 * h, 1e-6)
    min_imag_t = torch.tensor(min_imag, device=device, dtype=dtype)
    z_imag_abs = z_imag.abs()
    zero_mask = z_imag_abs <= 1e-12
    z_imag_default_t = torch.tensor(z_imag_default, device=device, dtype=dtype)
    z_real_default_t = torch.tensor(z_real_default, device=device, dtype=dtype)
    z_imag_final = torch.where(zero_mask, z_imag_default_t, z_imag_abs)
    z_imag_final = torch.clamp(z_imag_final, min=min_imag_t)
    z_real_final = torch.where(zero_mask, z_real_default_t, pos[2])

    pos = pos.clone()
    pos[2] = z_real_final
    params_out["position"] = pos
    params_out["z_imag"] = z_imag_final
    return params_out


def _compile_primitive(
    *,
    family: str,
    conductor_id: int,
    motif_index: int,
    spec: Any,
    device: torch.device,
    dtype: torch.dtype,
    layered_ctx: Dict[str, Any] | None,
    warnings: List[str],
) -> ImageBasisElement | None:
    type_name = _resolve_basis_type_name(family)
    pool = _candidate_pool(family, spec, device=device, dtype=dtype, conductor_id=conductor_id)
    if not pool:
        warnings.append(f"No candidates for family '{family}'")
        return None
    idx = motif_index % len(pool)
    pos, family_name = pool[idx]
    pos = _apply_layered_xy_override(pos, layered_ctx, device=device, dtype=dtype)
    elem = PointChargeBasis(
        {"position": pos, "z_imag": torch.zeros((), device=pos.device, dtype=pos.dtype)},
        type_name=type_name,
    )
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
