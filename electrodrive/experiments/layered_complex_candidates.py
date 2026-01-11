from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from electrodrive.experiments.layered_sampling import parse_layered_interfaces
from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock, StopProgram
from electrodrive.gfn.dsl.program import Program
from electrodrive.images.basis import (
    DCIMBranchCutImageBasis,
    DCIMPoleImageBasis,
    ImageBasisElement,
    PointChargeBasis,
    annotate_group_info,
)
from electrodrive.orchestration.parser import CanonicalSpec


@dataclass(frozen=True)
class LayeredComplexBoostConfig:
    enabled: bool = False
    programs: int = 0
    blocks_min: int = 2
    blocks_max: int = 4
    poles_min: int = 2
    poles_max: int = 8
    branches_min: int = 1
    branches_max: int = 3
    complex_terms_min: int = 2
    complex_terms_max: int = 6
    anchor_terms: int = 1
    xy_jitter: float = 0.05
    imag_min_scale: float = 1e-3
    imag_max_scale: float = 8.0
    log_cluster_std: float = 0.6

    @staticmethod
    def from_dict(raw: object) -> "LayeredComplexBoostConfig":
        if not isinstance(raw, dict):
            return LayeredComplexBoostConfig(enabled=bool(raw))
        return LayeredComplexBoostConfig(
            enabled=bool(raw.get("enabled", False)),
            programs=int(raw.get("programs", 0)),
            blocks_min=int(raw.get("blocks_min", 2)),
            blocks_max=int(raw.get("blocks_max", 4)),
            poles_min=int(raw.get("poles_min", 2)),
            poles_max=int(raw.get("poles_max", 8)),
            branches_min=int(raw.get("branches_min", 1)),
            branches_max=int(raw.get("branches_max", 3)),
            complex_terms_min=int(raw.get("complex_terms_min", 2)),
            complex_terms_max=int(raw.get("complex_terms_max", 6)),
            anchor_terms=int(raw.get("anchor_terms", 1)),
            xy_jitter=float(raw.get("xy_jitter", 0.05)),
            imag_min_scale=float(raw.get("imag_min_scale", 1e-3)),
            imag_max_scale=float(raw.get("imag_max_scale", 8.0)),
            log_cluster_std=float(raw.get("log_cluster_std", 0.6)),
        )


@dataclass(frozen=True)
class ManualCandidate:
    program: Program
    elements: List[ImageBasisElement]
    meta: Dict[str, object]


def _select_slab_layer(dielectrics: Sequence[dict]) -> Optional[dict]:
    slab_layer = None
    slab_thickness = None
    for layer in dielectrics:
        z_min = layer.get("z_min")
        z_max = layer.get("z_max")
        if z_min is None or z_max is None:
            continue
        try:
            thickness = float(z_max) - float(z_min)
        except Exception:
            continue
        if thickness <= 0:
            continue
        if layer.get("name") == "slab":
            return layer
        if slab_thickness is None or thickness < slab_thickness:
            slab_thickness = thickness
            slab_layer = layer
    return slab_layer


def _layer_region_id(z_val: float, top_z: float, bottom_z: float, tol: float = 1e-9) -> int:
    if z_val > top_z + tol:
        return 0
    if z_val < bottom_z - tol:
        return 2
    return 1


def _source_position(spec: CanonicalSpec) -> Optional[Tuple[float, float, float]]:
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


def _eps_ratio(spec: CanonicalSpec, slab: dict) -> float:
    eps_top = None
    for layer in getattr(spec, "dielectrics", None) or []:
        if not isinstance(layer, dict):
            continue
        if layer is slab:
            continue
        eps_top = layer.get("epsilon", layer.get("eps", None))
        break
    eps_slab = slab.get("epsilon", slab.get("eps", 1.0))
    try:
        eps_top_val = float(eps_top) if eps_top is not None else 1.0
    except Exception:
        eps_top_val = 1.0
    try:
        eps_slab_val = float(eps_slab)
    except Exception:
        eps_slab_val = 1.0
    if eps_top_val <= 0.0:
        eps_top_val = 1.0
    return abs(eps_slab_val / eps_top_val)


def _shift_from_interfaces(z_val: float, interfaces: Sequence[float], exclusion_radius: float) -> float:
    if exclusion_radius <= 0.0 or not interfaces:
        return z_val
    for plane in interfaces:
        if abs(z_val - plane) < exclusion_radius:
            direction = 1.0 if z_val >= plane else -1.0
            z_val = plane + direction * exclusion_radius
    return z_val


def _randint(gen: torch.Generator, low: int, high: int, device: torch.device) -> int:
    if high < low:
        high = low
    if high == low:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), generator=gen, device=device).item())


def _sample_imag_depths(
    count: int,
    h: float,
    eps_ratio: float,
    cfg: LayeredComplexBoostConfig,
    *,
    gen: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    min_imag = max(cfg.imag_min_scale * h, 1e-6)
    max_imag = max(cfg.imag_max_scale * h, min_imag * 10.0)
    u = torch.rand((count,), generator=gen, device=device, dtype=dtype)
    log_min = math.log(min_imag)
    log_max = math.log(max_imag)
    log_vals = log_min + (log_max - log_min) * u
    depths = torch.exp(log_vals)
    if cfg.log_cluster_std > 0.0:
        jitter = torch.randn((count,), generator=gen, device=device, dtype=dtype) * float(cfg.log_cluster_std)
        depths = depths * torch.exp(jitter)
    eps_scale = 1.0 + 0.25 * math.log1p(eps_ratio)
    depths = depths * eps_scale
    return torch.clamp(depths, min=min_imag)


def _make_position(
    base_xy: torch.Tensor,
    z_real: float,
    xy_jitter: float,
    *,
    gen: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if xy_jitter > 0.0:
        jitter = (torch.rand((2,), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * xy_jitter
        xy = base_xy + jitter
    else:
        xy = base_xy
    z = torch.tensor(float(z_real), device=device, dtype=dtype)
    return torch.stack([xy[0], xy[1], z], dim=0)


def build_layered_complex_candidates(
    spec: CanonicalSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    config: LayeredComplexBoostConfig,
    max_terms: int,
    domain_scale: float,
    exclusion_radius: float,
    allow_real_primitives: bool,
) -> List[ManualCandidate]:
    if not config.enabled or config.programs <= 0:
        return []
    if device.type != "cuda":
        raise ValueError("Layered complex boost requires CUDA device")
    if getattr(spec, "BCs", "") != "dielectric_interfaces":
        return []
    slab = _select_slab_layer(getattr(spec, "dielectrics", None) or [])
    if slab is None:
        return []
    source = _source_position(spec)
    if source is None:
        return []

    x0, y0, z0 = source
    top_z = float(slab.get("z_max", 0.0))
    bottom_z = float(slab.get("z_min", 0.0))
    h = float(top_z - bottom_z)
    if h <= 0.0:
        return []

    eps_ratio = _eps_ratio(spec, slab)
    interfaces = parse_layered_interfaces(spec)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    base_xy = torch.tensor([x0, y0], device=device, dtype=dtype)
    xy_jitter = float(config.xy_jitter) * float(domain_scale)

    out: List[ManualCandidate] = []
    max_terms = max(1, int(max_terms))
    programs = max(0, int(config.programs))
    for prog_idx in range(programs):
        elements: List[ImageBasisElement] = []
        nodes: List[AddPrimitiveBlock] = []
        motif = prog_idx * max_terms
        block_count = _randint(gen, config.blocks_min, config.blocks_max, device)
        block_count = max(1, block_count)

        dcim_poles = 0
        dcim_branches = 0
        dcim_blocks = 0

        for block_id in range(block_count):
            n_poles = _randint(gen, config.poles_min, config.poles_max, device)
            n_branches = _randint(gen, config.branches_min, config.branches_max, device)
            if n_poles <= 0 and n_branches <= 0:
                continue
            dcim_blocks += 1
            side_pick = torch.rand((), generator=gen, device=device).item()
            use_top = side_pick < 0.5
            if use_top:
                z_pole = 2.0 * top_z - z0 - 2.0 * block_id * h
                z_branch = 2.0 * top_z - z0 - (2.0 * block_id + 1.0) * h
            else:
                z_pole = 2.0 * bottom_z - z0 + 2.0 * block_id * h
                z_branch = 2.0 * bottom_z - z0 + (2.0 * block_id + 1.0) * h
            z_pole = _shift_from_interfaces(z_pole, interfaces, exclusion_radius)
            z_branch = _shift_from_interfaces(z_branch, interfaces, exclusion_radius)

            depths = _sample_imag_depths(
                n_poles + n_branches,
                h,
                eps_ratio,
                config,
                gen=gen,
                device=device,
                dtype=dtype,
            )
            if depths.numel() == 0:
                continue
            for idx in range(n_poles):
                pos = _make_position(base_xy, z_pole, xy_jitter, gen=gen, device=device, dtype=dtype)
                z_imag = depths[idx].view(())
                elem = DCIMPoleImageBasis({"position": pos, "z_imag": z_imag})
                region_id = _layer_region_id(float(z_pole), top_z, bottom_z)
                annotate_group_info(
                    elem,
                    conductor_id=region_id,
                    family_name="dcim_pole",
                    motif_index=motif,
                )
                info = getattr(elem, "_group_info", {})
                if isinstance(info, dict):
                    info["block_id"] = int(block_id)
                    info["block_kind"] = "dcim_pole"
                    setattr(elem, "_group_info", info)
                elements.append(elem)
                nodes.append(AddPrimitiveBlock(family_name="dcim_pole", conductor_id=region_id, motif_id=motif))
                motif += 1
                dcim_poles += 1
            for idx in range(n_branches):
                pos = _make_position(base_xy, z_branch, xy_jitter, gen=gen, device=device, dtype=dtype)
                z_imag = depths[n_poles + idx].view(())
                elem = DCIMBranchCutImageBasis({"position": pos, "z_imag": z_imag})
                region_id = _layer_region_id(float(z_branch), top_z, bottom_z)
                annotate_group_info(
                    elem,
                    conductor_id=region_id,
                    family_name="dcim_branch",
                    motif_index=motif,
                )
                info = getattr(elem, "_group_info", {})
                if isinstance(info, dict):
                    info["block_id"] = int(block_id)
                    info["block_kind"] = "dcim_branch"
                    setattr(elem, "_group_info", info)
                elements.append(elem)
                nodes.append(AddPrimitiveBlock(family_name="dcim_branch", conductor_id=region_id, motif_id=motif))
                motif += 1
                dcim_branches += 1

        n_complex = _randint(gen, config.complex_terms_min, config.complex_terms_max, device)
        for idx in range(n_complex):
            depth = _sample_imag_depths(1, h, eps_ratio, config, gen=gen, device=device, dtype=dtype)
            z_imag = depth[0].view(())
            z_real = 2.0 * top_z - z0 - (idx + 1) * h
            z_real = _shift_from_interfaces(z_real, interfaces, exclusion_radius)
            pos = _make_position(base_xy, z_real, xy_jitter, gen=gen, device=device, dtype=dtype)
            elem = PointChargeBasis({"position": pos, "z_imag": z_imag}, type_name="three_layer_images")
            region_id = _layer_region_id(float(z_real), top_z, bottom_z)
            annotate_group_info(
                elem,
                conductor_id=region_id,
                family_name="layered_complex",
                motif_index=motif,
            )
            elements.append(elem)
            nodes.append(AddPrimitiveBlock(family_name="layered_complex", conductor_id=region_id, motif_id=motif))
            motif += 1

        if allow_real_primitives and config.anchor_terms > 0:
            for idx in range(int(config.anchor_terms)):
                z_real = 2.0 * top_z - z0 - (idx + 1) * 0.5 * h
                z_real = _shift_from_interfaces(z_real, interfaces, exclusion_radius)
                pos = _make_position(base_xy, z_real, xy_jitter, gen=gen, device=device, dtype=dtype)
                elem = PointChargeBasis(
                    {"position": pos, "z_imag": torch.zeros((), device=device, dtype=dtype)},
                    type_name="three_layer_images",
                )
                region_id = _layer_region_id(float(z_real), top_z, bottom_z)
                annotate_group_info(
                    elem,
                    conductor_id=region_id,
                    family_name="layered_anchor",
                    motif_index=motif,
                )
                elements.append(elem)
                nodes.append(AddPrimitiveBlock(family_name="layered_anchor", conductor_id=region_id, motif_id=motif))
                motif += 1

        if not elements:
            continue
        if len(elements) > max_terms:
            elements = elements[:max_terms]
            nodes = nodes[:max_terms]
        nodes.append(StopProgram())

        meta = {
            "manual_candidate": True,
            "dcim_poles": int(dcim_poles),
            "dcim_branches": int(dcim_branches),
            "dcim_blocks": int(dcim_blocks),
            "program_index": int(prog_idx),
        }
        out.append(ManualCandidate(program=Program(nodes=tuple(nodes)), elements=elements, meta=meta))

    return out


__all__ = [
    "LayeredComplexBoostConfig",
    "ManualCandidate",
    "build_layered_complex_candidates",
]
