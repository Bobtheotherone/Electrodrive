from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml

from electrodrive.experiments.utils import (
    append_jsonl,
    assert_cuda_or_die,
    assert_cuda_tensor,
    ensure_dir,
    git_sha,
    git_status,
    seed_all,
    set_env,
    utc_timestamp,
    write_json,
    write_yaml,
)
from electrodrive.flows.types import FlowConfig
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.integration import GFlowNetProgramGenerator
from electrodrive.gfn.integration.gfn_basis_generator import _spec_metadata_from_spec
from electrodrive.gfn.integration.gfn_flow_generator import HybridGFlowFlowGenerator
from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock, StopProgram
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.rollout import SpecBatchItem, rollout_on_policy
from electrodrive.images.basis import (
    DCIMBranchCutImageBasis,
    DCIMPoleImageBasis,
    ImageBasisElement,
    PointChargeBasis,
    annotate_group_info,
)
from electrodrive.images.basis_dcim import DCIMBlockBasis
from electrodrive.images.geo_encoder import GeoEncoder
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.images.search import ImageSystem, assemble_basis_matrix, solve_sparse
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.experiments.reference_math import add_reference, stable_subtract_reference
from electrodrive.verify.oracle_backends.f0 import F0AnalyticOracleBackend
from electrodrive.verify.oracle_backends.f1_sommerfeld import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity
from electrodrive.verify.utils import normalize_dtype
from electrodrive.verify.utils import sha256_json, utc_now_iso
from electrodrive.verify.verifier import VerificationPlan, Verifier
from electrodrive.verify.gate_proxies import proxy_gateA, proxy_gateB, proxy_gateC, proxy_gateD
from electrodrive.experiments.layered_sampling import (
    parse_layered_interfaces,
    sample_layered_interior,
    sample_layered_interface_pairs,
)
from electrodrive.experiments.fast_proxy_metrics import (
    condition_ratio,
    far_field_ratio,
    interface_jump,
    log10_bucket,
)
from electrodrive.experiments.layered_complex_candidates import (
    LayeredComplexBoostConfig,
    build_layered_complex_candidates,
)
from electrodrive.experiments.preflight import (
    RunCounters,
    summarize_to_stdout,
    write_first_offender,
    write_preflight_report,
)
from electrodrive.experiments.vtrain_diagnostics import (
    build_vtrain_explosion_snapshot,
    write_vtrain_explosion_snapshot,
)


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"run_discovery.{run_dir.name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _format_preflight_summary(counters: RunCounters) -> str:
    return (
        "PREFLIGHT summary:"
        f" sampled={counters.sampled_programs_total}"
        f" compiled_ok={counters.compiled_ok}"
        f" empty_basis={counters.compiled_empty_basis}"
        f" compiled_failed={counters.compiled_failed}"
        f" solved_ok={counters.solved_ok}"
        f" fast_scored={counters.fast_scored}"
        f" verified_written={counters.verified_written}"
    )


@dataclass
class BackoffState:
    population_B: int
    bc_train: int
    bc_holdout: int
    interior_train: int
    interior_holdout: int


def _load_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping, got: {type(raw)}")
    return raw


def _resolve_checkpoint(config: Dict[str, Any], key: str, defaults: Sequence[Path]) -> Optional[Path]:
    user_path = (
        config.get("paths", {}).get(key)
        if isinstance(config.get("paths", {}), dict)
        else None
    )
    if user_path:
        path = Path(user_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"{key} checkpoint not found: {path}")
    for candidate in defaults:
        if candidate.is_file():
            return candidate
    return None


def _resolve_dtype(name: str) -> torch.dtype:
    key = (name or "fp32").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp64", "float64", "double"}:
        return torch.float64
    return torch.float32


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _set_perf_flags(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    use_tf32 = bool(model_cfg.get("use_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def should_early_exit(allow_not_ready: bool, ramp_abort: bool) -> bool:
    return bool(ramp_abort) and not bool(allow_not_ready)


def _make_run_dir(tag: str) -> Path:
    stamp = utc_timestamp()
    run_dir = Path("runs") / f"{stamp}_{tag}"
    ensure_dir(run_dir)
    for sub in ("artifacts/plots", "artifacts/checkpoints", "artifacts/certificates"):
        ensure_dir(run_dir / sub)
    return run_dir


def _env_report() -> Dict[str, Any]:
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "",
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_mem_gb": float(props.total_memory) / 1024**3,
    }


def _git_report() -> Dict[str, Any]:
    return {"git_sha": git_sha(), "git_status": git_status()}


class SpecSampler:
    def __init__(self, cfg: Dict[str, Any], seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        import random

        self._rng = random.Random(int(seed))

    def _rand_uniform(self, lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * self._rng.random())

    def sample(self) -> CanonicalSpec:
        family = str(self.cfg.get("family", "plane")).strip().lower()
        domain_scale = float(self.cfg.get("domain_scale", 1.0))

        if family in {"plane", "plane_point", "plane_halfspace"}:
            z0 = self._rand_uniform(*self._range("source_height_range", (0.2, 1.0)))
            x0 = self._rand_uniform(-0.2 * domain_scale, 0.2 * domain_scale)
            y0 = self._rand_uniform(-0.2 * domain_scale, 0.2 * domain_scale)
            spec = {
                "domain": {"bbox": [[-domain_scale, -domain_scale, -domain_scale], [domain_scale, domain_scale, domain_scale]]},
                "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
                "dielectrics": [],
                "charges": [{"type": "point", "q": 1.0, "charge": 1.0, "pos": [x0, y0, z0]}],
                "BCs": "Dirichlet",
                "symmetry": ["rot_z"],
                "queries": [],
            }
            return CanonicalSpec.from_json(spec)

        if family in {"layered_3layer", "three_layer"}:
            eps_lo, eps_hi = self._range("eps_range", (1.0, 10.0))
            h_lo, h_hi = self._range("thickness_range", (0.1, 0.5))
            z_lo, z_hi = self._range("source_height_range", (0.1, 1.0))
            eps2 = self._rand_uniform(eps_lo, eps_hi)
            h = self._rand_uniform(h_lo, h_hi)
            z0 = self._rand_uniform(z_lo, z_hi)
            x0 = self._rand_uniform(-0.2 * domain_scale, 0.2 * domain_scale)
            y0 = self._rand_uniform(-0.2 * domain_scale, 0.2 * domain_scale)
            spec = {
                "domain": {"bbox": [[-domain_scale, -domain_scale, -h - domain_scale], [domain_scale, domain_scale, domain_scale]]},
                "conductors": [],
                "dielectrics": [
                    {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": math.inf},
                    {"name": "slab", "epsilon": eps2, "z_min": -h, "z_max": 0.0},
                    {"name": "region3", "epsilon": 1.0, "z_min": -math.inf, "z_max": -h},
                ],
                "charges": [{"type": "point", "q": 1.0, "charge": 1.0, "pos": [x0, y0, z0]}],
                "BCs": "dielectric_interfaces",
                "symmetry": ["rot_z"],
                "queries": [],
            }
            return CanonicalSpec.from_json(spec)

        raise ValueError(f"Unsupported spec family: {family}")

    def _range(self, key: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
        raw = self.cfg.get(key, fallback)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return float(raw[0]), float(raw[1])
        return fallback


def _sample_points_gpu(
    spec: CanonicalSpec,
    *,
    n_boundary: int,
    n_interior: int,
    domain_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    min_z: float = 1e-3,
    exclusion_radius: float = 5e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    xy = (torch.rand((n_boundary, 2), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
    bc_points = torch.stack(
        [xy[:, 0], xy[:, 1], torch.zeros(n_boundary, device=device, dtype=dtype)],
        dim=1,
    )

    interior = _sample_interior_points(
        spec,
        n_interior=n_interior,
        domain_scale=domain_scale,
        device=device,
        dtype=dtype,
        gen=gen,
        min_z=min_z,
        exclusion_radius=exclusion_radius,
    )
    return bc_points, interior


def _sample_points_layered(
    spec: CanonicalSpec,
    *,
    n_boundary: int,
    n_interior: int,
    domain_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    exclusion_radius: float,
    interface_band: float,
    interface_delta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    interfaces = parse_layered_interfaces(spec)
    if not interfaces:
        return _sample_points_gpu(
            spec,
            n_boundary=n_boundary,
            n_interior=n_interior,
            domain_scale=domain_scale,
            device=device,
            dtype=dtype,
            seed=seed,
            exclusion_radius=exclusion_radius,
        )

    n_interfaces = max(1, len(interfaces))
    n_xy = max(1, n_boundary // (2 * n_interfaces))
    bc_up, bc_dn = sample_layered_interface_pairs(
        spec,
        n_xy,
        device=device,
        dtype=dtype,
        seed=seed,
        delta=interface_delta,
        domain_scale=domain_scale,
    )
    bc_points = torch.cat([bc_up, bc_dn], dim=0)
    if bc_points.shape[0] < n_boundary:
        extra_pairs = int(math.ceil((n_boundary - bc_points.shape[0]) / 2))
        extra_up, extra_dn = sample_layered_interface_pairs(
            spec,
            extra_pairs,
            device=device,
            dtype=dtype,
            seed=seed + 11,
            delta=interface_delta,
            domain_scale=domain_scale,
        )
        bc_points = torch.cat([bc_points, extra_up, extra_dn], dim=0)
    bc_points = bc_points[:n_boundary].contiguous()

    interior = sample_layered_interior(
        spec,
        n_interior,
        device=device,
        dtype=dtype,
        seed=seed,
        exclusion_radius=exclusion_radius,
        interface_band=interface_band,
        domain_scale=domain_scale,
    )
    return bc_points, interior


def _interface_eps_pairs(spec: CanonicalSpec) -> List[Tuple[float, float, float]]:
    dielectrics = getattr(spec, "dielectrics", None) or []
    interfaces: List[Tuple[float, float, float]] = []
    for layer in dielectrics:
        if "z_max" not in layer:
            continue
        z_int = float(layer["z_max"])
        eps_below = float(layer.get("epsilon", layer.get("eps", 1.0)))
        for other in dielectrics:
            if "z_min" not in other:
                continue
            z_lower = float(other["z_min"])
            if abs(z_lower - z_int) < 1e-6:
                eps_above = float(other.get("epsilon", other.get("eps", 1.0)))
                interfaces.append((z_int, eps_above, eps_below))
                break
    return interfaces


def _assemble_basis_normal_derivative(
    elements: Sequence[ImageBasisElement],
    points: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    if points.numel() == 0 or not elements:
        return torch.empty((points.shape[0], 0), device=points.device, dtype=points.dtype)
    cols: List[torch.Tensor] = []
    normal = normal.to(device=points.device, dtype=points.dtype)
    for elem in elements:
        pts = points.detach().clone().requires_grad_(True)
        vals = elem.potential(pts)
        grad = torch.autograd.grad(vals, pts, grad_outputs=torch.ones_like(vals), create_graph=False)[0]
        cols.append(torch.sum(grad * normal, dim=1))
    return torch.stack(cols, dim=1)


def _reference_normal_derivative(
    spec: CanonicalSpec,
    points: torch.Tensor,
    normal: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pts = points.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
    vals = compute_layered_reference_potential(spec, pts, device=device, dtype=dtype)
    grad = torch.autograd.grad(vals, pts, grad_outputs=torch.ones_like(vals), create_graph=False)[0]
    return torch.sum(grad * normal.to(device=device, dtype=dtype), dim=1).detach()


def _build_interface_constraint_data(
    spec: CanonicalSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    domain_scale: float,
    interface_delta: float,
    points_per_interface: int,
    weight: float,
    use_ref: bool,
) -> Optional[Dict[str, object]]:
    if points_per_interface <= 0 or weight <= 0.0:
        return None
    z_planes = parse_layered_interfaces(spec)
    if not z_planes:
        return None
    eps_pairs = _interface_eps_pairs(spec)
    if not eps_pairs:
        return None
    eps_map = {z: (eps_up, eps_dn) for z, eps_up, eps_dn in eps_pairs}
    pts_up, pts_dn = sample_layered_interface_pairs(
        spec,
        points_per_interface,
        device=device,
        dtype=dtype,
        seed=seed,
        delta=interface_delta,
        domain_scale=domain_scale,
    )
    if pts_up.numel() == 0:
        return None

    eps_up_list: List[float] = []
    eps_dn_list: List[float] = []
    for z_val in z_planes:
        eps_up, eps_dn = eps_map.get(z_val, (1.0, 1.0))
        eps_up_list.append(float(eps_up))
        eps_dn_list.append(float(eps_dn))
    eps_up_vec = torch.repeat_interleave(
        torch.tensor(eps_up_list, device=device, dtype=dtype),
        points_per_interface,
    )
    eps_dn_vec = torch.repeat_interleave(
        torch.tensor(eps_dn_list, device=device, dtype=dtype),
        points_per_interface,
    )
    n = min(pts_up.shape[0], eps_up_vec.shape[0])
    pts_up = pts_up[:n]
    pts_dn = pts_dn[:n]
    eps_up_vec = eps_up_vec[:n]
    eps_dn_vec = eps_dn_vec[:n]

    ref_phi = torch.zeros(n, device=device, dtype=dtype)
    ref_d = torch.zeros(n, device=device, dtype=dtype)
    if use_ref:
        ref_dtype = torch.float64
        ref_up = compute_layered_reference_potential(
            spec,
            pts_up.to(dtype=ref_dtype),
            device=device,
            dtype=ref_dtype,
        )
        ref_dn = compute_layered_reference_potential(
            spec,
            pts_dn.to(dtype=ref_dtype),
            device=device,
            dtype=ref_dtype,
        )
        ref_phi = (-(ref_up - ref_dn)).to(dtype=dtype)
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=ref_dtype)
        dref_up = _reference_normal_derivative(
            spec,
            pts_up,
            normal,
            device=device,
            dtype=ref_dtype,
        )
        dref_dn = _reference_normal_derivative(
            spec,
            pts_dn,
            normal,
            device=device,
            dtype=ref_dtype,
        )
        eps_up_ref = eps_up_vec.to(dtype=ref_dtype)
        eps_dn_ref = eps_dn_vec.to(dtype=ref_dtype)
        ref_d = (-(eps_up_ref * dref_up - eps_dn_ref * dref_dn)).to(dtype=dtype)

    return {
        "pts_up": pts_up,
        "pts_dn": pts_dn,
        "eps_up": eps_up_vec,
        "eps_dn": eps_dn_vec,
        "ref_phi": ref_phi,
        "ref_d": ref_d,
        "weight": float(weight),
    }


def _apply_interface_constraints(
    *,
    A_train: torch.Tensor,
    V_train: torch.Tensor,
    is_boundary: Optional[torch.Tensor],
    elements: Sequence[ImageBasisElement],
    constraint_data: Optional[Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if constraint_data is None or not elements:
        return A_train, V_train, is_boundary
    pts_up = constraint_data["pts_up"]
    pts_dn = constraint_data["pts_dn"]
    if pts_up.numel() == 0 or pts_dn.numel() == 0:
        return A_train, V_train, is_boundary

    A_up = assemble_basis_matrix(elements, pts_up)
    A_dn = assemble_basis_matrix(elements, pts_dn)
    A_phi = A_up - A_dn
    V_phi = constraint_data["ref_phi"].to(dtype=V_train.dtype)

    normal = torch.tensor([0.0, 0.0, 1.0], device=pts_up.device, dtype=pts_up.dtype)
    dA_up = _assemble_basis_normal_derivative(elements, pts_up, normal)
    dA_dn = _assemble_basis_normal_derivative(elements, pts_dn, normal)
    eps_up = constraint_data["eps_up"].to(dtype=dA_up.dtype).view(-1, 1)
    eps_dn = constraint_data["eps_dn"].to(dtype=dA_dn.dtype).view(-1, 1)
    A_d = eps_up * dA_up - eps_dn * dA_dn
    V_d = constraint_data["ref_d"].to(dtype=V_train.dtype)

    A_constraints = torch.cat([A_phi, A_d], dim=0)
    V_constraints = torch.cat([V_phi, V_d], dim=0)
    weight = float(constraint_data.get("weight", 1.0))
    if weight != 1.0:
        A_constraints = A_constraints * weight
        V_constraints = V_constraints * weight

    A_train = torch.cat([A_train, A_constraints], dim=0)
    V_train = torch.cat([V_train, V_constraints], dim=0)
    if is_boundary is not None:
        extra = torch.zeros(V_constraints.shape[0], device=is_boundary.device, dtype=torch.bool)
        is_boundary = torch.cat([is_boundary, extra], dim=0)
    return A_train, V_train, is_boundary


def _sample_interior_points(
    spec: CanonicalSpec,
    *,
    n_interior: int,
    domain_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
    min_z: float,
    exclusion_radius: float,
) -> torch.Tensor:
    if n_interior <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    attempts = 0
    needed = n_interior
    out: List[torch.Tensor] = []
    charges = getattr(spec, "charges", []) or []
    charge_pos = None
    if charges:
        charge_pos = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], device=device, dtype=dtype)

    while needed > 0 and attempts < 6:
        attempts += 1
        n_draw = max(needed * 2, 64)
        xyz = (torch.rand((n_draw, 3), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
        xyz[:, 2] = torch.rand((n_draw,), generator=gen, device=device, dtype=dtype) * domain_scale
        xyz[:, 2] = torch.clamp(xyz[:, 2], min=min_z)
        if charge_pos is not None:
            dist = torch.cdist(xyz, charge_pos)
            mask = torch.all(dist > exclusion_radius, dim=1)
            xyz = xyz[mask]
        if xyz.numel() == 0:
            continue
        take = min(needed, xyz.shape[0])
        out.append(xyz[:take])
        needed -= take

    if needed > 0:
        pad = torch.rand((needed, 3), generator=gen, device=device, dtype=dtype)
        pad = (pad - 0.5) * 2.0 * domain_scale
        pad[:, 2] = torch.rand((needed,), generator=gen, device=device, dtype=dtype) * domain_scale
        pad[:, 2] = torch.clamp(pad[:, 2], min=min_z)
        out.append(pad)
    return torch.cat(out, dim=0)


def _oracle_eval(
    backend: Any,
    spec: CanonicalSpec,
    points: torch.Tensor,
    *,
    dtype: torch.dtype,
    quantity: OracleQuantity = OracleQuantity.POTENTIAL,
) -> Tuple[torch.Tensor, Any]:
    query = OracleQuery(
        spec=spec.to_json(),
        points=points.to(device=points.device, dtype=dtype),
        quantity=quantity,
        requested_fidelity=backend.fidelity,
        device=str(points.device),
        dtype=normalize_dtype(dtype),
        cache_policy=CachePolicy.OFF,
        budget={},
    )
    result = backend.evaluate(query)
    if result.V is None:
        raise RuntimeError("Oracle returned no potential values.")
    assert_cuda_tensor(result.V, "oracle_V")
    return result.V, result


def _resample_invalid_oracle_targets(
    *,
    points: torch.Tensor,
    values: torch.Tensor,
    sample_fn: Callable[[int, int], torch.Tensor],
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    max_abs: float,
    max_attempts: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if points.numel() == 0 or values.numel() == 0 or max_attempts <= 0:
        return points, values, 0, 0
    pts = points.clone()
    vals = values.clone()
    total_nonfinite = 0
    total_extreme = 0
    for attempt in range(max_attempts + 1):
        nonfinite_mask = ~torch.isfinite(vals)
        if max_abs > 0.0:
            extreme_mask = torch.abs(vals) > max_abs
        else:
            extreme_mask = torch.zeros_like(nonfinite_mask)
        invalid_mask = nonfinite_mask | extreme_mask
        invalid_count = int(torch.count_nonzero(invalid_mask).item())
        if invalid_count == 0:
            break
        total_nonfinite += int(torch.count_nonzero(nonfinite_mask).item())
        total_extreme += int(torch.count_nonzero(extreme_mask).item())
        if attempt == max_attempts:
            raise RuntimeError("Oracle target resampling exhausted without producing finite targets.")
        new_points = sample_fn(invalid_count, seed + attempt + 1)
        new_values = eval_fn(new_points).to(dtype=vals.dtype)
        new_finite = torch.isfinite(new_values)
        if max_abs > 0.0:
            new_valid = new_finite & (torch.abs(new_values) <= max_abs)
        else:
            new_valid = new_finite
        if not torch.any(new_valid):
            continue
        invalid_idx = torch.nonzero(invalid_mask, as_tuple=False).view(-1)
        valid_idx = torch.nonzero(new_valid, as_tuple=False).view(-1)
        n_replace = min(int(invalid_idx.numel()), int(valid_idx.numel()))
        if n_replace > 0:
            replace_idx = invalid_idx[:n_replace]
            src_idx = valid_idx[:n_replace]
            pts[replace_idx] = new_points[src_idx]
            vals[replace_idx] = new_values[src_idx]
    return pts, vals, total_nonfinite, total_extreme


def _laplacian_fd(eval_fn: Any, pts: torch.Tensor, h: float) -> torch.Tensor:
    V0 = eval_fn(pts)
    lap = torch.zeros_like(V0)
    eye = torch.eye(3, device=pts.device, dtype=pts.dtype) * h
    for dim in range(3):
        offset = eye[dim].unsqueeze(0)
        plus = eval_fn(pts + offset)
        minus = eval_fn(pts - offset)
        lap = lap + (plus - 2.0 * V0 + minus) / (h * h)
    return lap


def _add_reference(pred: torch.Tensor, ref: Optional[torch.Tensor]) -> torch.Tensor:
    return add_reference(pred, ref)


def _proxy_fail_count(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> int:
    if not metrics:
        return 0
    fail = 0
    lap_tol = float(thresholds.get("laplacian_linf", 5e-3))
    status = metrics.get("proxy_gateA_status")
    if status == "fail":
        fail += 1
    elif status is None:
        linf = metrics.get("proxy_gateA_linf")
        l2 = metrics.get("proxy_gateA_l2")
        if linf is not None and float(linf) > lap_tol:
            fail += 1
        elif l2 is not None and float(l2) > lap_tol:
            fail += 1
    bc_tol = float(thresholds.get("bc_continuity", 5e-3))
    if float(metrics.get("proxy_gateB_max_v_jump", 0.0)) > bc_tol or float(
        metrics.get("proxy_gateB_max_d_jump", 0.0)
    ) > bc_tol:
        fail += 1
    slope_tol = float(thresholds.get("slope_tol", 0.15))
    far_slope = metrics.get("proxy_gateC_far_slope")
    near_slope = metrics.get("proxy_gateC_near_slope")
    if far_slope is not None and abs(float(far_slope) + 1.0) > slope_tol:
        fail += 1
    if near_slope is not None and abs(float(near_slope) + 1.0) > slope_tol:
        fail += 1
    if float(metrics.get("proxy_gateC_spurious_fraction", 0.0)) > 0.05:
        fail += 1
    stability_tol = float(thresholds.get("stability", 5e-2))
    if float(metrics.get("proxy_gateD_rel_change", 0.0)) > stability_tol:
        fail += 1
    return fail


def _proxy_fail_count_noA(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> int:
    if not metrics:
        return 0
    fail = 0
    bc_tol = float(thresholds.get("bc_continuity", 5e-3))
    if float(metrics.get("proxy_gateB_max_v_jump", 0.0)) > bc_tol or float(
        metrics.get("proxy_gateB_max_d_jump", 0.0)
    ) > bc_tol:
        fail += 1
    slope_tol = float(thresholds.get("slope_tol", 0.15))
    far_slope = metrics.get("proxy_gateC_far_slope")
    near_slope = metrics.get("proxy_gateC_near_slope")
    if far_slope is not None and abs(float(far_slope) + 1.0) > slope_tol:
        fail += 1
    if near_slope is not None and abs(float(near_slope) + 1.0) > slope_tol:
        fail += 1
    if float(metrics.get("proxy_gateC_spurious_fraction", 0.0)) > 0.05:
        fail += 1
    stability_tol = float(thresholds.get("stability", 5e-2))
    if float(metrics.get("proxy_gateD_rel_change", 0.0)) > stability_tol:
        fail += 1
    return fail


def _proxyA_effective_ratio(metrics: Dict[str, Any], cap: float, transform: str) -> float:
    a_raw = float(metrics.get("proxy_gateA_worst_ratio", float("inf")))
    if not math.isfinite(a_raw):
        a_raw = cap
    a_clamped = min(a_raw, cap)
    if transform == "logcap":
        return math.log10(1.0 + a_clamped)
    return a_clamped


PROXY_NONFINITE_PENALTY = 1e30


def _safe_proxy_float(value: Any) -> Tuple[float, bool]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0, True
    if not math.isfinite(val):
        return 0.0, True
    return val, False


def _proxy_score_with_sanitized(
    metrics: Dict[str, Any],
    *,
    a_weight: float,
    a_cap: float,
    a_transform: str,
) -> Tuple[float, bool]:
    if not metrics:
        return float(PROXY_NONFINITE_PENALTY), True
    a_eff = _proxyA_effective_ratio(metrics, a_cap, a_transform)
    b_v, b_v_bad = _safe_proxy_float(metrics.get("proxy_gateB_max_v_jump", None))
    b_d, b_d_bad = _safe_proxy_float(metrics.get("proxy_gateB_max_d_jump", None))
    if b_v_bad or b_d_bad:
        return float(PROXY_NONFINITE_PENALTY), True
    far_slope, far_bad = _safe_proxy_float(metrics.get("proxy_gateC_far_slope", None))
    near_slope, near_bad = _safe_proxy_float(metrics.get("proxy_gateC_near_slope", None))
    spurious, spurious_bad = _safe_proxy_float(metrics.get("proxy_gateC_spurious_fraction", None))
    d, d_bad = _safe_proxy_float(metrics.get("proxy_gateD_rel_change", None))
    if far_bad or near_bad or spurious_bad or d_bad:
        return float(PROXY_NONFINITE_PENALTY), True
    c = abs(far_slope + 1.0) + abs(near_slope + 1.0) + spurious * 10.0
    score = a_weight * a_eff + max(b_v, b_d) + c + d
    if not math.isfinite(score):
        return float(PROXY_NONFINITE_PENALTY), True
    return float(score), False


def _proxy_score(
    metrics: Dict[str, Any],
    *,
    a_weight: float,
    a_cap: float,
    a_transform: str,
) -> float:
    score, _ = _proxy_score_with_sanitized(
        metrics,
        a_weight=a_weight,
        a_cap=a_cap,
        a_transform=a_transform,
    )
    return score


def build_proxy_stability_points(
    bc_hold: torch.Tensor,
    interior_hold: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    n = max(1, int(n_points))
    if bc_hold.numel() == 0:
        return interior_hold[:n]
    if interior_hold.numel() == 0:
        return bc_hold[:n]
    n_bc = min(bc_hold.shape[0], n // 2)
    n_in = min(interior_hold.shape[0], n - n_bc)
    if n_bc == 0:
        return interior_hold[:n_in]
    if n_in == 0:
        return bc_hold[:n_bc]
    return torch.cat([bc_hold[:n_bc], interior_hold[:n_in]], dim=0)


def _fast_weights(
    A: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    *,
    normalize: bool = True,
    max_abs_A: Optional[float] = None,
    max_abs_b: Optional[float] = None,
    fp64_threshold: float = 1e6,
) -> torch.Tensor:
    if A.numel() == 0:
        return torch.zeros((0,), device=A.device, dtype=A.dtype)
    k = A.shape[1]
    if k == 0:
        return torch.zeros((0,), device=A.device, dtype=A.dtype)
    use_fp64 = False
    if max_abs_b is None:
        max_abs_b = _tensor_absmax(b)
    if max_abs_b > fp64_threshold:
        use_fp64 = True
    if not use_fp64:
        if max_abs_A is None:
            max_abs_A = _tensor_absmax(A)
        if max_abs_A > fp64_threshold:
            use_fp64 = True
    if use_fp64 and A.dtype != torch.float64:
        A_work = A.to(dtype=torch.float64)
        b_work = b.to(dtype=torch.float64)
    else:
        A_work = A
        b_work = b
    if normalize:
        A_scaled, col_norms = _scale_columns(A_work)
    else:
        A_scaled = A_work
        col_norms = torch.ones((k,), device=A_work.device, dtype=A_work.dtype)
    ata = A_scaled.transpose(0, 1).matmul(A_scaled)
    ata = ata + reg * torch.eye(k, device=A_work.device, dtype=A_work.dtype)
    atb = A_scaled.transpose(0, 1).matmul(b_work)
    try:
        w_scaled = torch.linalg.solve(ata, atb)
    except RuntimeError:
        try:
            w_scaled = torch.linalg.lstsq(A_scaled, b_work).solution
        except RuntimeError:
            return torch.zeros((k,), device=A.device, dtype=A.dtype)
    weights = w_scaled / col_norms
    if use_fp64 and weights.dtype != A.dtype:
        weights = weights.to(dtype=A.dtype)
    return weights


def _scale_columns(A: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.numel() == 0:
        return A, torch.ones((A.shape[1],), device=A.device, dtype=A.dtype)
    col_norms = torch.linalg.norm(A, dim=0).clamp_min(eps)
    return A / col_norms, col_norms


def _nonfinite_count(t: torch.Tensor) -> int:
    return int(torch.count_nonzero(~torch.isfinite(t)).item())


def _validate_weights(weights: torch.Tensor) -> Tuple[bool, str]:
    if not torch.is_tensor(weights):
        return False, "weights_not_tensor"
    if weights.numel() == 0:
        return False, "weights_empty"
    if torch.is_complex(weights):
        return False, "weights_complex"
    if not weights.is_floating_point():
        return False, "weights_nonfloat"
    if not torch.isfinite(weights).all().item():
        return False, "weights_nonfinite"
    return True, ""


def _weights_serializable(weights: torch.Tensor) -> Tuple[Optional[List[float]], str]:
    ok, reason = _validate_weights(weights)
    if not ok:
        return None, reason
    return weights.detach().cpu().tolist(), ""


def _tensor_absmax(t: Optional[torch.Tensor]) -> float:
    if t is None or t.numel() == 0:
        return float("nan")
    finite = torch.isfinite(t)
    if not torch.any(finite):
        return float("nan")
    vals = torch.abs(t[finite])
    return float(torch.max(vals).item())


def _mean_abs(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    vals = torch.abs(x)
    if vals.dtype != torch.float64:
        vals = vals.double()
    return float(torch.mean(vals).item())


def _mean_safe(x: torch.Tensor, *, fail_value: float = float("inf")) -> float:
    if x.numel() == 0:
        return 0.0
    mean = torch.mean(x)
    if torch.isfinite(mean).item():
        return float(mean.item())
    mean64 = torch.mean(x.double())
    if torch.isfinite(mean64).item():
        return float(mean64.item())
    return float(fail_value)


def _laplacian_denom(oracle_in_mean_abs: float, oracle_bc_mean_abs: float) -> float:
    return max(float(oracle_in_mean_abs), 1e-6 * float(oracle_bc_mean_abs), 1e-12)


HOLDOUT_FAIL_VALUE = 1e30


def _finite(val: Any) -> bool:
    if torch.is_tensor(val):
        if val.numel() == 0:
            return True
        if val.numel() != 1:
            return bool(torch.isfinite(val).all().item())
        return bool(torch.isfinite(val).item())
    try:
        return math.isfinite(float(val))
    except Exception:
        return False


def _finite_or_fail(val: Any, fail_value: float) -> float:
    if not _finite(val):
        return float(fail_value)
    if torch.is_tensor(val):
        if val.numel() == 0:
            return float(fail_value)
        return float(val.item())
    return float(val)


def _tensor_all_finite(t: Optional[torch.Tensor]) -> bool:
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return True
    return bool(torch.isfinite(t).all().item())


def _sanitize_metric_block(
    values: Dict[str, Any],
    *,
    fail_value: float,
) -> Tuple[bool, Dict[str, float], List[str]]:
    nonfinite = []
    sanitized: Dict[str, float] = {}
    for key, val in values.items():
        if _finite(val):
            sanitized[key] = _finite_or_fail(val, fail_value)
        else:
            nonfinite.append(key)
            sanitized[key] = float(fail_value)
    return len(nonfinite) == 0, sanitized, nonfinite


def _holdout_partition_flags(n_boundary: int, n_interior: int) -> List[str]:
    flags: List[str] = []
    if n_boundary <= 0:
        flags.append("holdout_boundary_empty")
    if n_interior <= 0:
        flags.append("holdout_interior_empty")
    return flags


def _timed_cuda(fn: Any, *args: Any, warmup: int = 1, repeat: int = 3) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    for _ in range(max(0, warmup)):
        _ = fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(max(1, repeat)):
        start.record()
        _ = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(sum(times) / len(times))


def _complexity(program: Any, elements: Sequence[ImageBasisElement]) -> Dict[str, int]:
    return {"n_nodes": int(len(getattr(program, "nodes", []) or [])), "n_terms": int(len(elements))}


def _element_type_hist(elements: Sequence[ImageBasisElement]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for elem in elements:
        name = elem.__class__.__name__
        hist[name] = hist.get(name, 0) + 1
    return dict(sorted(hist.items()))


def _count_complex_terms(
    elements: Sequence[ImageBasisElement],
    *,
    device: torch.device,
    thresh: float = 1e-6,
) -> int:
    if not elements:
        return 0
    z_vals: List[torch.Tensor] = []
    for elem in elements:
        z_imag = elem.params.get("z_imag")
        if z_imag is None:
            continue
        if torch.is_tensor(z_imag):
            z = z_imag.to(device=device).view(())
        else:
            try:
                z = torch.tensor(float(z_imag), device=device)
            except Exception:
                continue
        z_vals.append(z)
    if not z_vals:
        return 0
    z_stack = torch.stack(z_vals)
    return int((z_stack.abs() > thresh).sum().item())


def _dcim_stats(elements: Sequence[ImageBasisElement]) -> Dict[str, int]:
    dcim_poles = 0
    dcim_branches = 0
    dcim_blocks = 0
    block_ids: set[int] = set()
    for elem in elements:
        if isinstance(elem, DCIMPoleImageBasis):
            dcim_poles += 1
        elif isinstance(elem, DCIMBranchCutImageBasis):
            dcim_branches += 1
        elif isinstance(elem, DCIMBlockBasis):
            dcim_blocks += 1
        else:
            elem_type = getattr(elem, "type", "")
            if isinstance(elem_type, str) and elem_type.startswith("dcim_block"):
                dcim_blocks += 1
        info = getattr(elem, "_group_info", None)
        if isinstance(info, dict) and "block_id" in info:
            try:
                block_ids.add(int(info.get("block_id", 0)))
            except Exception:
                pass
    dcim_terms = dcim_poles + dcim_branches + dcim_blocks
    if dcim_terms > 0 and not block_ids:
        block_ids.add(0)
    return {
        "dcim_poles": int(dcim_poles),
        "dcim_branches": int(dcim_branches),
        "dcim_blocks": int(len(block_ids)),
        "dcim_terms": int(dcim_terms),
    }


def _max_abs_imag_depth(
    elements: Sequence[ImageBasisElement],
    *,
    device: torch.device,
) -> float:
    max_val = torch.zeros((), device=device, dtype=torch.float32)
    for elem in elements:
        z_imag = elem.params.get("z_imag")
        if z_imag is not None:
            if torch.is_tensor(z_imag):
                z_val = torch.abs(z_imag.to(device=device).reshape(-1))
                if z_val.numel() > 0:
                    max_val = torch.maximum(max_val, torch.max(z_val))
            else:
                try:
                    z_val = abs(float(z_imag))
                    max_val = torch.maximum(max_val, torch.tensor(z_val, device=device))
                except Exception:
                    pass
        if isinstance(elem, DCIMBlockBasis) and getattr(elem, "images", None):
            try:
                depths = torch.tensor(
                    [img.depth for img in elem.images], device=device, dtype=torch.complex64
                )
                imag = torch.abs(depths.imag)
                if imag.numel() > 0:
                    max_val = torch.maximum(max_val, torch.max(imag))
            except Exception:
                pass
    if not torch.isfinite(max_val):
        return float("nan")
    return float(max_val.item())


def _bucket_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 9:
        return "6-9"
    return "10+"


def _candidate_signature(dcim_stats: Dict[str, int], imag_bucket: str) -> str:
    block_count = dcim_stats.get("dcim_blocks", 0)
    pole_bucket = _bucket_count(dcim_stats.get("dcim_poles", 0))
    return f"b{block_count}_p{pole_bucket}_i{imag_bucket}"


def _speed_proxy(
    n_terms: int,
    dcim_terms: int,
    dcim_blocks: int,
    *,
    dcim_term_cost: float,
    dcim_block_cost: float,
) -> float:
    base = float(n_terms)
    return base + float(dcim_terms) * float(dcim_term_cost) + float(dcim_blocks) * float(dcim_block_cost)


def _update_hist(hist: Dict[str, int], key: object, n: int = 1) -> None:
    k = str(key)
    hist[k] = hist.get(k, 0) + int(n)


def _sample_sphere_points(
    n: int,
    radius: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    vec = torch.randn((n, 3), generator=gen, device=device, dtype=dtype)
    norms = torch.linalg.norm(vec, dim=1, keepdim=True).clamp_min(1e-12)
    vec = vec / norms
    return vec * float(radius)


def _violates_interface_exclusion(
    elements: Sequence[ImageBasisElement],
    interfaces: Sequence[float],
    exclusion_radius: float,
    *,
    device: torch.device,
) -> bool:
    if exclusion_radius <= 0.0 or not interfaces:
        return False
    z_vals: List[torch.Tensor] = []
    for elem in elements:
        pos = elem.params.get("position")
        if pos is None:
            continue
        if torch.is_tensor(pos):
            z_vals.append(pos.to(device=device).reshape(-1)[2])
    if not z_vals:
        return False
    z_stack = torch.stack(z_vals)
    planes = torch.tensor(list(interfaces), device=device, dtype=z_stack.dtype)
    dists = torch.abs(z_stack[:, None] - planes[None, :])
    return bool(torch.any(dists < exclusion_radius).item())


def _perturb_elements(
    elements: Sequence[ImageBasisElement],
    sigma: float,
    *,
    device: torch.device,
) -> Optional[List[ImageBasisElement]]:
    perturbed: List[ImageBasisElement] = []
    try:
        for elem in elements:
            params: Dict[str, torch.Tensor] = {}
            for k, v in elem.params.items():
                if not torch.is_tensor(v):
                    params[k] = v
                    continue
                if torch.is_floating_point(v) or torch.is_complex(v):
                    noise = torch.randn_like(v) * sigma
                    params[k] = (v + noise).to(device=device)
                else:
                    params[k] = v.clone().to(device=device)
            new_elem = elem.__class__(params)
            info = getattr(elem, "_group_info", None)
            if isinstance(info, dict):
                setattr(new_elem, "_group_info", dict(info))
            perturbed.append(new_elem)
        return perturbed
    except Exception:
        return None


def _has_dcim_block(elements: Sequence[ImageBasisElement]) -> bool:
    for elem in elements:
        elem_type = getattr(elem, "type", "")
        if isinstance(elem_type, str) and elem_type.startswith("dcim_block"):
            return True
        class_name = elem.__class__.__name__.lower()
        if "dcim" in class_name:
            return True
        if hasattr(elem, "block") or hasattr(elem, "certificate"):
            return True
    return False


def _dcim_used_as_best(
    best_rel_metrics: Optional[Dict[str, Any]],
    best_rel_elements: Optional[Sequence[ImageBasisElement]],
) -> bool:
    if best_rel_metrics and best_rel_metrics.get("is_dcim_block_baseline"):
        return not bool(best_rel_metrics.get("holdout_nonfinite"))
    if best_rel_elements and _has_dcim_block(best_rel_elements):
        if best_rel_metrics and best_rel_metrics.get("holdout_nonfinite"):
            return False
        return True
    return False


def _build_dcim_block_elements(
    spec: CanonicalSpec,
    *,
    device: torch.device,
    max_blocks: int,
    cache_path: Path,
) -> List[ImageBasisElement]:
    missing: List[str] = []
    dielectrics = getattr(spec, "dielectrics", None) or []
    if not dielectrics:
        missing.append("spec.dielectrics")
    charges = getattr(spec, "charges", None) or []
    point_charges = [c for c in charges if isinstance(c, dict) and c.get("type") == "point"]
    if not point_charges:
        missing.append("spec.charges[type=point]")
    else:
        pos0 = point_charges[0].get("pos", None)
        if pos0 is None or len(pos0) < 3:
            missing.append("spec.charges[].pos")
        if point_charges[0].get("charge", point_charges[0].get("q", None)) is None:
            missing.append("spec.charges[].charge/q")
    if missing:
        raise RuntimeError(
            "DCIM block baseline missing required fields: " + ", ".join(sorted(set(missing)))
        )

    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("DCIM block baseline requires CUDA device for compilation.")

    try:
        from electrodrive.layers import (
            DCIMCompilerConfig,
            SpectralKernelSpec,
            compile_dcim,
            layerstack_from_spec,
        )
    except Exception as exc:
        raise RuntimeError(f"DCIM block baseline imports failed: {exc}") from exc

    stack = layerstack_from_spec(spec)
    kernel = SpectralKernelSpec(
        source_region=0,
        obs_region=0,
        component="potential",
        bc_kind="dielectric_interfaces",
    )
    pos0 = point_charges[0].get("pos", [0.0, 0.0, 0.2])
    x_src = float(pos0[0])
    y_src = float(pos0[1])
    z_src = float(pos0[2])
    q_src = float(point_charges[0].get("charge", point_charges[0].get("q", 1.0)))

    cfg = DCIMCompilerConfig(
        k_min=0.05,
        k_mid=2.0,
        k_max=6.0,
        n_low=64,
        n_mid=64,
        n_high=0,
        log_low=False,
        log_high=False,
        vf_enabled=False,
        vf_for_images=False,
        exp_fit_enabled=True,
        exp_fit_requires_uniform_grid=True,
        exp_N=6,
        spectral_tol=0.3,
        spatial_tol=0.2,
        sample_points=[(0.3, 0.6), (0.5, 1.0), (0.2, 0.6)],
        cache_enabled=True,
        cache_path=cache_path,
        device=dev,
        dtype=torch.complex128,
        runtime_eval_mode="image_only",
        source_z=z_src,
        source_charge=q_src,
        source_pos=(x_src, y_src, z_src),
    )

    with torch.inference_mode():
        block = compile_dcim(stack, kernel, cfg)

    if not block.certificate.stable:
        raise RuntimeError("DCIM block baseline compile produced unstable certificate.")

    from electrodrive.images.basis_dcim import dcim_basis_from_block

    elems = dcim_basis_from_block(block)
    if not elems:
        raise RuntimeError("DCIM block baseline produced no basis elements.")
    for idx, elem in enumerate(elems):
        annotate_group_info(
            elem,
            conductor_id=0,
            family_name="dcim_block",
            motif_index=idx,
        )
        info = getattr(elem, "_group_info", None)
        if isinstance(info, dict):
            info["block_id"] = 0
            info["block_kind"] = "dcim_block_baseline"
            setattr(elem, "_group_info", info)

    max_blocks = max(0, int(max_blocks))
    if max_blocks <= 0:
        return []
    return elems[:max_blocks]


def _program_to_json(program: Any) -> List[Dict[str, Any]]:
    nodes = getattr(program, "nodes", []) or []
    return [n.to_dict() for n in nodes]


def _baseline_programs_for_spec(spec: CanonicalSpec) -> List[Program]:
    conductors = getattr(spec, "conductors", None) or []
    charges = getattr(spec, "charges", None) or []
    if not conductors or not charges:
        return []
    if conductors[0].get("type") != "plane":
        return []
    if charges[0].get("type") != "point":
        return []
    nodes = (
        AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),
        AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=1),
        StopProgram(),
    )
    return [Program(nodes=nodes)]


def _seed_layered_templates(
    spec: CanonicalSpec,
    *,
    max_steps: int,
    count: int = 8,
    allow_real_primitives: bool = True,
) -> List[Program]:
    if not allow_real_primitives:
        return []
    if getattr(spec, "BCs", "") != "dielectric_interfaces":
        return []
    layers = getattr(spec, "dielectrics", None) or []
    if len(layers) < 3:
        return []
    max_terms = max_steps - 1  # reserve one slot for StopProgram.
    if max_terms < 4:
        return []
    n_terms = min(12, max_terms)
    if n_terms < 8:
        n_terms = max_terms
    families = ("three_layer_images", "three_layer_mirror", "three_layer_slab", "three_layer_tail")
    out: List[Program] = []
    for seed_idx in range(max(1, min(count, 8))):
        nodes: List[Any] = []
        for node_idx in range(n_terms):
            family = families[(seed_idx + node_idx) % len(families)]
            nodes.append(
                AddPrimitiveBlock(
                    family_name=family,
                    conductor_id=0,
                    motif_id=seed_idx * n_terms + node_idx,
                )
            )
        nodes.append(StopProgram())
        out.append(Program(nodes=tuple(nodes)))
    return out


def _candidate_to_record(
    gen: int,
    rank: int,
    program: Any,
    elements: Sequence[ImageBasisElement],
    weights: torch.Tensor,
    metrics: Dict[str, Any],
    score: float,
) -> Dict[str, Any]:
    return {
        "generation": int(gen),
        "rank": int(rank),
        "score": float(score),
        "program": _program_to_json(program),
        "elements": [e.serialize() for e in elements],
        "weights": weights.detach().cpu().tolist(),
        "metrics": metrics,
    }


def _select_topk(scores: Sequence[float], k: int) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    return order[: max(0, min(k, len(order)))]


def _diversity_select(indices: Sequence[int], signatures: Dict[int, str], k: int) -> List[int]:
    if not indices:
        return []
    buckets: Dict[str, List[int]] = {}
    for idx in indices:
        sig = signatures.get(idx, "none")
        buckets.setdefault(sig, []).append(idx)
    selected: List[int] = []
    while buckets and len(selected) < k:
        for sig in list(buckets.keys()):
            if not buckets[sig]:
                buckets.pop(sig, None)
                continue
            selected.append(buckets[sig].pop(0))
            if len(selected) >= k:
                break
        buckets = {sig: vals for sig, vals in buckets.items() if vals}
    return selected[: max(0, min(k, len(selected)))]


def _oracle_for_name(name: str) -> OracleFidelity:
    key = (name or "fast").strip().lower()
    if key in {"mid", "f1"}:
        return OracleFidelity.F1
    if key in {"hi", "f2"}:
        return OracleFidelity.F2
    return OracleFidelity.F0


def _build_oracle_backend(fidelity: OracleFidelity, spec: CanonicalSpec) -> Any:
    if fidelity == OracleFidelity.F1:
        backend = F1SommerfeldOracleBackend()
        if backend.can_handle(OracleQuery(spec=spec.to_json(), points=torch.empty(0, 3, device="cuda"), quantity=OracleQuantity.POTENTIAL, requested_fidelity=fidelity, device="cuda", dtype="float32", cache_policy=CachePolicy.OFF, budget={})):
            return backend
    return F0AnalyticOracleBackend()


def run_discovery(config_path: Path, *, debug: bool = False) -> int:
    cfg = _load_config(config_path)
    seed = int(cfg.get("seed", 0))
    seed_all(seed)

    assert_cuda_or_die()
    set_env("EDE_DEVICE", "cuda")
    set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.autograd.set_detect_anomaly(True)

    _set_perf_flags(cfg)

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    tag = str(run_cfg.get("tag", "discovery_v0")).strip() or "discovery_v0"
    preflight_mode_raw = str(run_cfg.get("preflight_mode", "")).strip().lower()
    if not preflight_mode_raw:
        preflight_enabled = bool(run_cfg.get("preflight_enabled", False))
        preflight_mode = "full" if preflight_enabled else "off"
    else:
        if preflight_mode_raw not in {"off", "lite", "full"}:
            raise ValueError(f"Unknown preflight_mode: {preflight_mode_raw}")
        preflight_mode = preflight_mode_raw
    preflight_enabled = preflight_mode != "off"
    preflight_full = preflight_mode == "full"
    preflight_lite = preflight_mode == "lite"
    preflight_out = str(run_cfg.get("preflight_out", "preflight.json")).strip() or "preflight.json"
    run_dir = _make_run_dir(tag)
    run_logger = _configure_run_logger(run_dir)
    run_logger.info("RUN START tag=%s preflight_mode=%s", tag, preflight_mode)
    dcim_cache_path = run_dir / "artifacts" / "dcim_cache.jsonl"

    write_json(run_dir / "env.json", _env_report())
    write_json(run_dir / "git.json", _git_report())
    write_yaml(run_dir / "config.yaml", cfg)

    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best.jsonl"

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    structure_cfg = (
        model_cfg.get("structure_policy", {})
        if isinstance(model_cfg.get("structure_policy", {}), dict)
        else {}
    )
    max_steps = int(structure_cfg.get("max_steps", 64))
    proposal_dtype = _resolve_dtype(str(model_cfg.get("dtype", "bf16")))
    if proposal_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        proposal_dtype = torch.float16
    param_cfg = model_cfg.get("param_sampler", {}) if isinstance(model_cfg.get("param_sampler", {}), dict) else {}
    param_name = str(param_cfg.get("name", "flow")).strip().lower()
    use_param_sampler = param_name not in {"none", "static", "gfn"}
    set_env("EDE_FLOW_COMPILE", "1" if use_param_sampler and bool(model_cfg.get("compile", True)) else "")

    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    gfn_ckpt = _resolve_checkpoint(
        cfg,
        "gfn_checkpoint",
        [
            Path(paths_cfg.get("gfn_checkpoint", "")),
            Path("artifacts/step10_gfn_flow_smoke/gfn_ckpt.pt"),
            Path("runs/step10_gfn_flow_smoke/run_1766285538/gfn.pt"),
        ],
    )
    flow_ckpt = None
    if use_param_sampler:
        flow_ckpt = _resolve_checkpoint(
            cfg,
            "flow_checkpoint",
            [
                Path(paths_cfg.get("flow_checkpoint", "")),
                Path("artifacts/step10_gfn_flow_smoke/flow_ckpt.pt"),
                Path("runs/step10_gfn_flow_smoke/run_1766285538/flow.pt"),
            ],
        )
    if gfn_ckpt is None:
        raise RuntimeError("Missing GFN checkpoint; set paths.gfn_checkpoint in config.")
    if use_param_sampler and flow_ckpt is None:
        raise RuntimeError("Missing flow checkpoint; set paths.flow_checkpoint in config.")

    device = torch.device("cuda")
    if use_param_sampler:
        flow_max_ast_len = _coerce_optional_int(param_cfg.get("max_ast_len"))
        flow_max_tokens = _coerce_optional_int(param_cfg.get("max_tokens"))
        if flow_max_ast_len is None or flow_max_ast_len < max_steps:
            flow_max_ast_len = max_steps
        if flow_max_tokens is None or flow_max_tokens < max_steps:
            flow_max_tokens = max_steps
        flow_latent_clip = _coerce_optional_float(param_cfg.get("latent_clip"))
        flow_cfg = FlowConfig(
            n_steps=int(param_cfg.get("steps", 4)),
            solver=str(param_cfg.get("solver", "euler")),
            temperature=float(param_cfg.get("temperature", 1.0)),
            latent_clip=flow_latent_clip,
            dtype=str(param_cfg.get("dtype", "bf16")),
            max_ast_len=flow_max_ast_len,
            max_tokens=flow_max_tokens,
        )
        gfn = HybridGFlowFlowGenerator(
            checkpoint_path=str(gfn_ckpt),
            flow_checkpoint_path=str(flow_ckpt),
            flow_config=flow_cfg,
            allow_random_flow=False,
            device=device,
            dtype=proposal_dtype,
            max_steps=max_steps,
        )
    else:
        gfn = GFlowNetProgramGenerator(
            checkpoint_path=str(gfn_ckpt),
            device=device,
            dtype=proposal_dtype,
            max_steps=max_steps,
        )
    if getattr(gfn, "env", None) is not None:
        if int(getattr(gfn.env, "max_length", 0)) < max_steps:
            gfn.env.max_length = int(max_steps)
        min_stop = _coerce_optional_int(structure_cfg.get("min_length_for_stop"))
        if min_stop is not None:
            gfn.env.min_length_for_stop = max(1, min_stop)

    spec_cfg = cfg.get("spec", {}) if isinstance(cfg.get("spec", {}), dict) else {}
    sampler = SpecSampler(spec_cfg, seed)

    spec_dim = int(getattr(gfn.policy.config, "spec_dim", 8))
    encoder_name = str(model_cfg.get("geom_encoder", {}).get("name", "simple")).strip().lower()
    if encoder_name in {"simple", "mlp"}:
        geo_encoder = SimpleGeoEncoder(latent_dim=spec_dim)
    else:
        geo_encoder = GeoEncoder(hidden_dim=spec_dim, n_layers=int(model_cfg.get("geom_encoder", {}).get("layers", 4)))
    geo_encoder = geo_encoder.to(device=device, dtype=torch.float32)

    oracle_cfg = cfg.get("oracle", {}) if isinstance(cfg.get("oracle", {}), dict) else {}
    fast_fidelity = _oracle_for_name(str(oracle_cfg.get("fast", {}).get("name", "fast")))
    mid_fidelity = _oracle_for_name(str(oracle_cfg.get("mid", {}).get("name", "mid")))
    fast_dtype = _resolve_dtype(str(oracle_cfg.get("fast", {}).get("dtype", "fp32")))
    mid_dtype = _resolve_dtype(str(oracle_cfg.get("mid", {}).get("dtype", "fp32")))

    solver_cfg = cfg.get("solver", {}) if isinstance(cfg.get("solver", {}), dict) else {}
    solver_name = str(solver_cfg.get("name", "differentiable_lasso")).strip().lower()
    if solver_name == "differentiable_lasso":
        solver_mode = "implicit_lasso"
    elif solver_name == "differentiable_grouplasso":
        solver_mode = "implicit_grouplasso"
    else:
        solver_mode = "implicit_lasso"

    reward_cfg = cfg.get("reward", {}) if isinstance(cfg.get("reward", {}), dict) else {}
    w_bc = float(reward_cfg.get("w_bc", 1.0))
    w_pde = float(reward_cfg.get("w_pde", 0.5))
    w_complexity = float(reward_cfg.get("w_complexity", 0.05))
    w_latency = float(reward_cfg.get("w_latency", 0.25))
    w_stability = float(reward_cfg.get("w_stability", 0.25))
    stability_sigma = float(reward_cfg.get("stability_perturb_sigma", 1e-3))

    points_cfg = cfg.get("points", {}) if isinstance(cfg.get("points", {}), dict) else {}
    train_target_abs_max = float(run_cfg.get("train_target_abs_max", 1e10))
    train_target_resample_attempts = int(run_cfg.get("train_target_resample_attempts", 4))
    if train_target_resample_attempts < 0:
        train_target_resample_attempts = 0

    backoff = BackoffState(
        population_B=int(run_cfg.get("population_B", 512)),
        bc_train=int(points_cfg.get("bc_train", 256)),
        bc_holdout=int(points_cfg.get("bc_holdout", 256)),
        interior_train=int(points_cfg.get("interior_train", 256)),
        interior_holdout=int(points_cfg.get("interior_holdout", 256)),
    )
    if debug:
        backoff.population_B = min(backoff.population_B, 64)
        backoff.bc_train = min(backoff.bc_train, 256)
        backoff.bc_holdout = min(backoff.bc_holdout, 256)
        backoff.interior_train = min(backoff.interior_train, 256)
        backoff.interior_holdout = min(backoff.interior_holdout, 256)

    fixed_spec = bool(run_cfg.get("fixed_spec", False))
    fixed_spec_index = int(run_cfg.get("fixed_spec_index", 0))
    if fixed_spec_index < 0:
        fixed_spec_index = 0
    generations = int(run_cfg.get("generations", 2))
    topK_fast = int(run_cfg.get("topK_fast", 8))
    topk_mid = int(run_cfg.get("topk_mid", 2))
    topk_final = int(run_cfg.get("topk_final", 2))
    score_microbatch = int(run_cfg.get("score_microbatch", 32))
    if score_microbatch <= 0:
        score_microbatch = 1
    cache_hold_matrices = bool(run_cfg.get("cache_hold_matrices", False))
    use_hi_oracle = bool(run_cfg.get("use_hi_oracle", False))
    use_dcim_block = bool(run_cfg.get("use_dcim_block", False))
    dcim_block_max = int(run_cfg.get("dcim_block_max", 0))
    if dcim_block_max < 0:
        dcim_block_max = 0
    dcim_block_weight = float(run_cfg.get("dcim_block_weight", 1.0))
    ramp_check = bool(run_cfg.get("ramp_check", False))
    allow_not_ready = bool(run_cfg.get("allow_not_ready", False))
    ramp_patience = int(run_cfg.get("ramp_patience_gens", 3))
    ramp_min_rel_improve = float(run_cfg.get("ramp_min_rel_improvement", 0.10))
    use_reference_potential = bool(run_cfg.get("use_reference_potential", False))
    use_gate_proxies = bool(run_cfg.get("use_gate_proxies", False))
    use_gateA_proxy = bool(run_cfg.get("use_gateA_proxy", use_gate_proxies))
    proxyA_transform = str(run_cfg.get("proxyA_transform", "logcap")).strip().lower()
    if not proxyA_transform:
        proxyA_transform = "logcap"
    proxyA_cap = float(run_cfg.get("proxyA_cap", 1e6))
    if not math.isfinite(proxyA_cap) or proxyA_cap <= 0.0:
        proxyA_cap = 1e6
    proxyA_weight = float(run_cfg.get("proxyA_weight", 1.0))
    if not math.isfinite(proxyA_weight):
        proxyA_weight = 1.0
    proxy_ranking_mode = str(run_cfg.get("proxy_ranking_mode", "balanced")).strip().lower()
    if proxy_ranking_mode not in {"balanced", "a_first"}:
        proxy_ranking_mode = "balanced"
    layered_sampling = bool(run_cfg.get("layered_sampling", True))
    layered_exclusion_radius = float(run_cfg.get("layered_exclusion_radius", 5e-3))
    layered_interface_delta = float(run_cfg.get("layered_interface_delta", 5e-3))
    layered_interface_band = float(run_cfg.get("layered_interface_band", layered_interface_delta))
    layered_stability_delta = float(run_cfg.get("layered_stability_delta", 1e-2))
    layered_prefer_dcim = bool(run_cfg.get("layered_prefer_dcim", True))
    layered_allow_real_primitives = bool(run_cfg.get("layered_allow_real_primitives", False))
    complex_boost_cfg = LayeredComplexBoostConfig.from_dict(run_cfg.get("layered_complex_boost", {}))
    fast_proxy_cfg = run_cfg.get("fast_proxy", {}) if isinstance(run_cfg.get("fast_proxy", {}), dict) else {}
    fast_proxy_enabled = bool(fast_proxy_cfg.get("enabled", False))
    fast_proxy_n_far = int(fast_proxy_cfg.get("n_far", 24))
    fast_proxy_n_interface = int(fast_proxy_cfg.get("n_interface", 24))
    fast_proxy_near_radius = float(fast_proxy_cfg.get("near_radius", 0.5))
    fast_proxy_far_radius = float(fast_proxy_cfg.get("far_radius", 12.0))
    fast_proxy_far_weight = float(fast_proxy_cfg.get("far_weight", 0.0))
    fast_proxy_interface_weight = float(fast_proxy_cfg.get("interface_weight", 0.0))
    fast_proxy_cond_weight = float(fast_proxy_cfg.get("cond_weight", 0.0))
    fast_proxy_speed_weight = float(fast_proxy_cfg.get("speed_weight", 0.0))
    fast_proxy_dcim_term_cost = float(fast_proxy_cfg.get("dcim_term_cost", 1.5))
    fast_proxy_dcim_block_cost = float(fast_proxy_cfg.get("dcim_block_cost", 2.5))
    fast_proxy_far_target = float(fast_proxy_cfg.get("far_target", 0.4))
    fast_proxy_far_max = float(fast_proxy_cfg.get("far_max_ratio", 4.0))
    fast_proxy_cond_target = float(fast_proxy_cfg.get("cond_target", 5e3))
    fast_proxy_interface_max = float(fast_proxy_cfg.get("interface_max_jump", 0.0))
    fast_proxy_cond_max = float(fast_proxy_cfg.get("cond_max", 0.0))
    fast_proxy_fail_hard = bool(fast_proxy_cfg.get("fail_hard", False))
    dcim_diversity = bool(run_cfg.get("dcim_diversity", False))
    diversity_guard = bool(run_cfg.get("diversity_guard", True))
    layered_enforce_interface_constraints = bool(run_cfg.get("layered_enforce_interface_constraints", False))
    layered_interface_constraint_weight = float(run_cfg.get("layered_interface_constraint_weight", 1.0))
    if not math.isfinite(layered_interface_constraint_weight) or layered_interface_constraint_weight <= 0.0:
        layered_interface_constraint_weight = 0.0
    layered_interface_constraint_points = int(run_cfg.get("layered_interface_constraint_points", 64))
    if layered_interface_constraint_points < 0:
        layered_interface_constraint_points = 0
    refine_enabled = bool(run_cfg.get("refine_enabled", False))
    refine_steps = int(run_cfg.get("refine_steps", 12))
    refine_lr = float(run_cfg.get("refine_lr", 5e-2))
    refine_opt = str(run_cfg.get("refine_opt", "adam")).strip().lower()
    refine_max_terms = int(run_cfg.get("refine_max_terms", 32))
    refine_targets = str(run_cfg.get("refine_targets", "holdout")).strip().lower()
    if refine_steps <= 0:
        refine_enabled = False
    if refine_opt not in {"adam", "lbfgs"}:
        refine_opt = "adam"
    if refine_targets not in {"holdout", "train"}:
        refine_targets = "holdout"
    sanity_threshold = run_cfg.get("sanity_bc_threshold", None)
    if sanity_threshold is not None:
        sanity_threshold = float(sanity_threshold)
    sanity_force_baseline = bool(run_cfg.get("sanity_force_baseline", False))
    best_mean_bc: Optional[float] = None
    best_rel_bc: Optional[float] = None
    best_rel_in: Optional[float] = None
    best_rel_elements: Optional[Sequence[ImageBasisElement]] = None
    best_rel_metrics: Optional[Dict[str, Any]] = None
    best_rel_in_metrics: Optional[Dict[str, Any]] = None
    best_rel_lap: Optional[float] = None
    best_rel_lap_metrics: Optional[Dict[str, Any]] = None
    ramp_best_rel_bc: Optional[float] = None
    ramp_best_rel_lap: Optional[float] = None
    last_improve_gen = 0
    per_gen_rel_bc: List[float] = []
    per_gen_rel_in: List[float] = []
    per_gen_rel_lap: List[float] = []
    per_gen_empty_frac: List[float] = []
    spec_hashes: List[str] = []
    best_records: List[Dict[str, Any]] = []
    refine_attempted_total = 0
    refine_improved_total = 0
    best_refined_rel_bc: Optional[float] = None
    best_refined_rel_lap: Optional[float] = None
    best_dcim_rel_bc: Optional[float] = None
    best_dcim_rel_lap: Optional[float] = None
    best_dcim_score: Optional[float] = None
    dcim_block_cache: Dict[str, List[ImageBasisElement]] = {}

    verifier = Verifier(out_root=run_dir / "artifacts" / "certificates")
    verify_plan = VerificationPlan()
    verify_plan.start_fidelity = fast_fidelity
    verify_plan.allow_f2 = bool(use_hi_oracle)
    verify_plan.oracle_budget = dict(verify_plan.oracle_budget)
    verify_plan.oracle_budget["allow_cpu_fallback"] = False
    ramp_abort = False
    config_hash = sha256_json(cfg)
    def _init_preflight_entry(gen_idx: int) -> Dict[str, Any]:
        return {
            "gen": int(gen_idx),
            "counters": RunCounters().as_dict(),
            "fraction_complex_candidates": 0.0,
            "fraction_dcim_candidates": 0.0,
        }

    preflight_counters = RunCounters() if preflight_enabled else None
    per_gen_preflight: List[Dict[str, Any]] = []
    if preflight_enabled:
        per_gen_preflight = [_init_preflight_entry(gen_idx) for gen_idx in range(generations)]
    gen_counters: Optional[RunCounters] = None
    first_offender_written = False
    v_train_snapshot_written = False
    holdout_offender_logged = False
    holdout_stats: Dict[str, Any] = {}
    weights_reject_reasons: set[str] = set()
    holdout_reject_reasons: set[str] = set()
    baseline_backend_name: Optional[str] = None
    dcim_pole_hist: Dict[str, int] = {}
    dcim_block_hist: Dict[str, int] = {}
    max_imag_hist: Dict[str, int] = {}
    max_weight_hist: Dict[str, int] = {}
    cond_ratio_hist: Dict[str, int] = {}
    low_dcim_streak = 0
    low_complex_streak = 0

    def _build_preflight_extra() -> Dict[str, Any]:
        assert preflight_counters is not None
        device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
        device_cap = (
            ".".join(str(v) for v in torch.cuda.get_device_capability(device))
            if torch.cuda.is_available()
            else "cpu"
        )
        nonfinite_total = max(1, preflight_counters.nonfinite_pred_total)
        extra = {
            "timestamp": utc_now_iso(),
            "tag": tag,
            "config_hash": config_hash,
            "git_sha": git_sha(),
            "device_name": device_name,
            "device_capability": device_cap,
            "preflight_mode": preflight_mode,
            "preflight_out": preflight_out,
            "train_target_abs_max": float(train_target_abs_max),
            "train_target_resample_attempts": int(train_target_resample_attempts),
            "nonfinite_pred_fraction": float(preflight_counters.nonfinite_pred_count) / float(nonfinite_total),
            "per_gen": per_gen_preflight,
        }
        candidate_total = max(1, int(preflight_counters.compiled_ok))
        extra["fraction_complex_candidates"] = float(preflight_counters.complex_candidates) / float(candidate_total)
        extra["fraction_dcim_candidates"] = float(preflight_counters.dcim_candidates) / float(candidate_total)
        extra["dcim_pole_count_hist"] = dict(sorted(dcim_pole_hist.items(), key=lambda x: x[0]))
        extra["dcim_block_count_hist"] = dict(sorted(dcim_block_hist.items(), key=lambda x: x[0]))
        extra["max_abs_imag_depth_hist"] = dict(sorted(max_imag_hist.items(), key=lambda x: x[0]))
        extra["max_abs_weight_hist"] = dict(sorted(max_weight_hist.items(), key=lambda x: x[0]))
        extra["condition_ratio_hist"] = dict(sorted(cond_ratio_hist.items(), key=lambda x: x[0]))
        extra["baseline_speed_backend_name"] = baseline_backend_name or "unknown"
        if weights_reject_reasons:
            extra["weights_reject_reasons"] = sorted(weights_reject_reasons)
        if holdout_reject_reasons:
            extra["holdout_reject_reasons"] = sorted(holdout_reject_reasons)
        return extra

    def _write_preflight_snapshot() -> None:
        if preflight_counters is None:
            return
        write_preflight_report(run_dir, preflight_counters, _build_preflight_extra())

    if preflight_enabled:
        _write_preflight_snapshot()
        run_logger.info("PREFLIGHT heartbeat initialized: %s", preflight_out)

    def _count_preflight(key: str, n: int = 1) -> None:
        if preflight_counters is None:
            return
        preflight_counters.add(key, n)
        if gen_counters is not None:
            gen_counters.add(key, n)

    def _maybe_write_first_offender(
        *,
        gen_idx: int,
        program_idx: int,
        program: Any,
        elements: Sequence[ImageBasisElement],
        A_train: Optional[torch.Tensor],
        V_train: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        pred_hold: Optional[torch.Tensor],
        reason: str,
        extra_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal first_offender_written
        if preflight_counters is None or first_offender_written:
            return
        stats = {
            "A_train_absmax": _tensor_absmax(A_train),
            "V_train_absmax": _tensor_absmax(V_train),
            "weights_absmax": _tensor_absmax(weights),
            "pred_hold_absmax": _tensor_absmax(pred_hold),
        }
        if extra_stats:
            stats.update(extra_stats)
        payload = {
            "gen": int(gen_idx),
            "program_idx": int(program_idx),
            "reason": str(reason),
            "program_repr": repr(program),
            "program": _program_to_json(program),
            "element_types": _element_type_hist(elements),
            "flags": {
                "use_reference_potential": bool(use_ref),
                "use_gate_proxies": bool(use_gate_proxies),
                "layered_prefer_dcim": bool(layered_prefer_dcim),
            },
            "stats": stats,
        }
        if write_first_offender(run_dir, payload):
            first_offender_written = True

    def _reject_invalid_weights(
        *,
        gen_idx: int,
        program_idx: int,
        program: Any,
        elements: Sequence[ImageBasisElement],
        A_train: Optional[torch.Tensor],
        V_train: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        pred_hold: Optional[torch.Tensor],
        reason: str,
    ) -> None:
        if preflight_counters is not None:
            _count_preflight("solved_failed")
            if torch.is_tensor(weights):
                _count_preflight("weights_total", int(weights.numel()))
                if reason == "weights_nonfinite":
                    _count_preflight("weights_nonfinite_count", _nonfinite_count(weights))
                if weights.numel() == 0:
                    _count_preflight("weights_empty")
        weights_reject_reasons.add(reason)
        _maybe_write_first_offender(
            gen_idx=gen_idx,
            program_idx=program_idx,
            program=program,
            elements=elements,
            A_train=A_train,
            V_train=V_train,
            weights=weights,
            pred_hold=pred_hold,
            reason=reason,
        )

    def _log_holdout_nonfinite(
        *,
        gen_idx: int,
        program_idx: int,
        program: Any,
        elements: Sequence[ImageBasisElement],
        A_train: Optional[torch.Tensor],
        V_train: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        pred_hold: Optional[torch.Tensor],
        V_hold: Optional[torch.Tensor],
        V_hold_mid: Optional[torch.Tensor],
        is_boundary_hold: Optional[torch.Tensor],
        lap: Optional[torch.Tensor],
        denom_in: float,
        denom_lap: float,
        reason: str,
    ) -> None:
        nonlocal holdout_offender_logged
        nonlocal holdout_stats
        holdout_reject_reasons.add(reason)
        if preflight_counters is None or holdout_offender_logged:
            return
        holdout_offender_logged = True

        pred_in = None
        oracle_in = None
        diff_in = None
        if pred_hold is not None and is_boundary_hold is not None:
            pred_in = pred_hold[~is_boundary_hold]
        if V_hold_mid is not None and is_boundary_hold is not None:
            oracle_in = V_hold_mid[~is_boundary_hold]
        elif V_hold is not None and is_boundary_hold is not None:
            oracle_in = V_hold[~is_boundary_hold]
        if pred_in is not None and oracle_in is not None:
            diff_in = pred_in - oracle_in

        def _nonfinite_or_zero(t: Optional[torch.Tensor]) -> int:
            if t is None or not torch.is_tensor(t) or t.numel() == 0:
                return 0
            return _nonfinite_count(t)

        stats = {
            "pred_in_nonfinite_count": _nonfinite_or_zero(pred_in),
            "oracle_in_nonfinite_count": _nonfinite_or_zero(oracle_in),
            "diff_in_nonfinite_count": _nonfinite_or_zero(diff_in),
            "lap_nonfinite_count": _nonfinite_or_zero(lap),
            "pred_in_absmax": _tensor_absmax(pred_in),
            "oracle_in_absmax": _tensor_absmax(oracle_in),
            "diff_in_absmax": _tensor_absmax(diff_in),
            "lap_absmax": _tensor_absmax(lap),
            "denom_in": float(denom_in),
            "denom_lap": float(denom_lap),
        }
        if holdout_stats:
            stats.update(holdout_stats)
        run_logger.warning(
            "NONFINITE HOLDOUT: gen=%s idx=%s reason=%s pred_in_nonfinite=%s oracle_in_nonfinite=%s "
            "diff_in_nonfinite=%s lap_nonfinite=%s denom_in=%.3e denom_lap=%.3e",
            gen_idx,
            program_idx,
            reason,
            stats["pred_in_nonfinite_count"],
            stats["oracle_in_nonfinite_count"],
            stats["diff_in_nonfinite_count"],
            stats["lap_nonfinite_count"],
            stats["denom_in"],
            stats["denom_lap"],
        )
        _maybe_write_first_offender(
            gen_idx=gen_idx,
            program_idx=program_idx,
            program=program,
            elements=elements,
            A_train=A_train,
            V_train=V_train,
            weights=weights,
            pred_hold=pred_hold,
            reason=reason,
            extra_stats=stats,
        )

    fixed_spec_obj: Optional[CanonicalSpec] = None
    if fixed_spec:
        for _ in range(fixed_spec_index + 1):
            fixed_spec_obj = sampler.sample()
        if fixed_spec_obj is None:
            raise RuntimeError("fixed_spec enabled but no spec could be sampled")

    for gen in range(generations):
        start_gen = time.perf_counter()
        if preflight_enabled:
            gen_counters = RunCounters()
        else:
            gen_counters = None
        if fixed_spec and fixed_spec_obj is not None:
            spec = fixed_spec_obj
        else:
            spec = sampler.sample()
        spec_hash = sha256_json(spec.to_json())
        spec_hashes.append(spec_hash)
        domain_scale = float(spec_cfg.get("domain_scale", 1.0))
        seed_gen = seed + gen * 13
        is_layered = getattr(spec, "BCs", "") == "dielectric_interfaces"
        interface_planes = parse_layered_interfaces(spec) if is_layered else []
        use_ref = bool(use_reference_potential and is_layered)
        prefer_dcim = bool(layered_prefer_dcim and is_layered)
        allow_real_primitives = bool(layered_allow_real_primitives or not prefer_dcim)
        if prefer_dcim and not use_param_sampler and not layered_allow_real_primitives:
            allow_real_primitives = True
        spec_meta = _spec_metadata_from_spec(
            spec,
            extra_overrides={"allow_real_primitives": allow_real_primitives},
        )

        torch.cuda.synchronize()
        retry = True
        attempt = 0
        while retry:
            attempt += 1
            retry = False
            try:
                if is_layered and layered_sampling:
                    bc_train, interior_train = _sample_points_layered(
                        spec,
                        n_boundary=backoff.bc_train,
                        n_interior=backoff.interior_train,
                        domain_scale=domain_scale,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 1,
                        exclusion_radius=layered_exclusion_radius,
                        interface_band=layered_interface_band,
                        interface_delta=layered_interface_delta,
                    )
                    bc_hold, interior_hold = _sample_points_layered(
                        spec,
                        n_boundary=backoff.bc_holdout,
                        n_interior=backoff.interior_holdout,
                        domain_scale=domain_scale,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 7,
                        exclusion_radius=layered_exclusion_radius,
                        interface_band=layered_interface_band,
                        interface_delta=layered_interface_delta,
                    )
                else:
                    bc_train, interior_train = _sample_points_gpu(
                        spec,
                        n_boundary=backoff.bc_train,
                        n_interior=backoff.interior_train,
                        domain_scale=domain_scale,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 1,
                    )
                    bc_hold, interior_hold = _sample_points_gpu(
                        spec,
                        n_boundary=backoff.bc_holdout,
                        n_interior=backoff.interior_holdout,
                        domain_scale=domain_scale,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 7,
                    )

                assert_cuda_tensor(bc_train, "bc_train")
                assert_cuda_tensor(interior_train, "interior_train")
                assert_cuda_tensor(bc_hold, "bc_hold")
                assert_cuda_tensor(interior_hold, "interior_hold")

                oracle_fast = _build_oracle_backend(fast_fidelity, spec)
                oracle_mid = _build_oracle_backend(mid_fidelity, spec)
                if baseline_backend_name is None:
                    baseline_backend_name = oracle_fast.__class__.__name__

                V_bc_train, _ = _oracle_eval(oracle_fast, spec, bc_train, dtype=fast_dtype)
                V_in_train, _ = _oracle_eval(oracle_fast, spec, interior_train, dtype=fast_dtype)
                V_bc_hold, _ = _oracle_eval(oracle_fast, spec, bc_hold, dtype=fast_dtype)
                V_in_hold, _ = _oracle_eval(oracle_fast, spec, interior_hold, dtype=fast_dtype)
                V_bc_train = V_bc_train.to(dtype=bc_train.dtype)
                V_in_train = V_in_train.to(dtype=interior_train.dtype)
                V_bc_hold = V_bc_hold.to(dtype=bc_hold.dtype)
                V_in_hold = V_in_hold.to(dtype=interior_hold.dtype)

                V_bc_hold_mid, _ = _oracle_eval(oracle_mid, spec, bc_hold, dtype=mid_dtype)
                V_in_hold_mid, _ = _oracle_eval(oracle_mid, spec, interior_hold, dtype=mid_dtype)
                V_bc_hold_mid = V_bc_hold_mid.to(dtype=bc_hold.dtype)
                V_in_hold_mid = V_in_hold_mid.to(dtype=interior_hold.dtype)
                def _sample_train_boundary(n: int, seed_val: int) -> torch.Tensor:
                    if is_layered and layered_sampling:
                        bc_pts, _ = _sample_points_layered(
                            spec,
                            n_boundary=n,
                            n_interior=0,
                            domain_scale=domain_scale,
                            device=device,
                            dtype=torch.float32,
                            seed=seed_val,
                            exclusion_radius=layered_exclusion_radius,
                            interface_band=layered_interface_band,
                            interface_delta=layered_interface_delta,
                        )
                    else:
                        bc_pts, _ = _sample_points_gpu(
                            spec,
                            n_boundary=n,
                            n_interior=0,
                            domain_scale=domain_scale,
                            device=device,
                            dtype=torch.float32,
                            seed=seed_val,
                        )
                    return bc_pts

                def _sample_train_interior(n: int, seed_val: int) -> torch.Tensor:
                    if is_layered and layered_sampling:
                        _, in_pts = _sample_points_layered(
                            spec,
                            n_boundary=0,
                            n_interior=n,
                            domain_scale=domain_scale,
                            device=device,
                            dtype=torch.float32,
                            seed=seed_val,
                            exclusion_radius=layered_exclusion_radius,
                            interface_band=layered_interface_band,
                            interface_delta=layered_interface_delta,
                        )
                    else:
                        _, in_pts = _sample_points_gpu(
                            spec,
                            n_boundary=0,
                            n_interior=n,
                            domain_scale=domain_scale,
                            device=device,
                            dtype=torch.float32,
                            seed=seed_val,
                        )
                    return in_pts

                def _eval_fast(pts: torch.Tensor) -> torch.Tensor:
                    vals, _ = _oracle_eval(oracle_fast, spec, pts, dtype=fast_dtype)
                    return vals.to(dtype=pts.dtype)

                try:
                    bc_train, V_bc_train, bc_resample_nonfinite, bc_resample_extreme = (
                        _resample_invalid_oracle_targets(
                            points=bc_train,
                            values=V_bc_train,
                            sample_fn=_sample_train_boundary,
                            eval_fn=_eval_fast,
                            max_abs=train_target_abs_max,
                            max_attempts=train_target_resample_attempts,
                            seed=seed_gen + 31,
                        )
                    )
                    interior_train, V_in_train, in_resample_nonfinite, in_resample_extreme = (
                        _resample_invalid_oracle_targets(
                            points=interior_train,
                            values=V_in_train,
                            sample_fn=_sample_train_interior,
                            eval_fn=_eval_fast,
                            max_abs=train_target_abs_max,
                            max_attempts=train_target_resample_attempts,
                            seed=seed_gen + 37,
                        )
                    )
                except RuntimeError as exc:
                    X_train = torch.cat([bc_train, interior_train], dim=0)
                    V_train = torch.cat([V_bc_train, V_in_train], dim=0)
                    payload = build_vtrain_explosion_snapshot(
                        spec,
                        X_train,
                        V_train,
                        None,
                        layered_reference_enabled=use_ref,
                        reference_subtracted_for_fit=bool(use_ref),
                        nan_to_num_applied=False,
                        clamp_applied=False,
                        seed=seed_gen,
                        gen=gen,
                        program_idx=None,
                    )
                    payload["reason"] = "oracle_resample_exhausted"
                    write_vtrain_explosion_snapshot(run_dir, payload)
                    raise RuntimeError(str(exc)) from exc

                if preflight_counters is not None:
                    total_nonfinite = int(bc_resample_nonfinite + in_resample_nonfinite)
                    total_extreme = int(bc_resample_extreme + in_resample_extreme)
                    if total_nonfinite > 0:
                        _count_preflight("oracle_nonfinite_resample_count", total_nonfinite)
                    if total_extreme > 0:
                        _count_preflight("oracle_extreme_resample_count", total_extreme)
                bc_hold_mask = torch.isfinite(V_bc_hold) & torch.isfinite(V_bc_hold_mid)
                in_hold_mask = torch.isfinite(V_in_hold) & torch.isfinite(V_in_hold_mid)
                if not torch.all(bc_hold_mask):
                    bc_hold = bc_hold[bc_hold_mask]
                    V_bc_hold = V_bc_hold[bc_hold_mask]
                    V_bc_hold_mid = V_bc_hold_mid[bc_hold_mask]
                if not torch.all(in_hold_mask):
                    interior_hold = interior_hold[in_hold_mask]
                    V_in_hold = V_in_hold[in_hold_mask]
                    V_in_hold_mid = V_in_hold_mid[in_hold_mask]
                holdout_resampled = False
                if interior_hold.shape[0] == 0:
                    for attempt in range(4):
                        seed_resample = seed_gen + 101 + attempt
                        if is_layered and layered_sampling:
                            interior_candidate = sample_layered_interior(
                                spec,
                                backoff.interior_holdout,
                                device=device,
                                dtype=torch.float32,
                                seed=seed_resample,
                                exclusion_radius=layered_exclusion_radius,
                                interface_band=layered_interface_band,
                                domain_scale=domain_scale,
                            )
                        else:
                            _, interior_candidate = _sample_points_gpu(
                                spec,
                                n_boundary=0,
                                n_interior=backoff.interior_holdout,
                                domain_scale=domain_scale,
                                device=device,
                                dtype=torch.float32,
                                seed=seed_resample,
                            )
                        if interior_candidate.numel() == 0:
                            continue
                        V_in_candidate, _ = _oracle_eval(
                            oracle_fast,
                            spec,
                            interior_candidate,
                            dtype=fast_dtype,
                        )
                        V_in_candidate_mid, _ = _oracle_eval(
                            oracle_mid,
                            spec,
                            interior_candidate,
                            dtype=mid_dtype,
                        )
                        V_in_candidate = V_in_candidate.to(dtype=interior_candidate.dtype)
                        V_in_candidate_mid = V_in_candidate_mid.to(dtype=interior_candidate.dtype)
                        in_mask = torch.isfinite(V_in_candidate) & torch.isfinite(V_in_candidate_mid)
                        if not torch.all(in_mask):
                            interior_candidate = interior_candidate[in_mask]
                            V_in_candidate = V_in_candidate[in_mask]
                            V_in_candidate_mid = V_in_candidate_mid[in_mask]
                        if interior_candidate.shape[0] == 0:
                            continue
                        interior_hold = interior_candidate
                        V_in_hold = V_in_candidate
                        V_in_hold_mid = V_in_candidate_mid
                        holdout_resampled = True
                        break
                if bc_train.shape[0] + interior_train.shape[0] == 0:
                    raise RuntimeError("No finite training targets after filtering.")
                if bc_hold.shape[0] + interior_hold.shape[0] == 0:
                    raise RuntimeError("No finite holdout targets after filtering.")
                holdout_boundary = int(bc_hold.shape[0])
                holdout_interior = int(interior_hold.shape[0])
                holdout_total = holdout_boundary + holdout_interior
                holdout_partition_flags = _holdout_partition_flags(holdout_boundary, holdout_interior)
                if holdout_partition_flags:
                    holdout_reject_reasons.update(holdout_partition_flags)
                if preflight_counters is not None:
                    _count_preflight("holdout_total", holdout_total)
                    _count_preflight("holdout_boundary_total", holdout_boundary)
                    _count_preflight("holdout_interior_total", holdout_interior)
                    if holdout_boundary == 0:
                        _count_preflight("holdout_boundary_empty_count")
                    if holdout_interior == 0:
                        _count_preflight("holdout_interior_empty_count")
                oracle_bc_mean_abs = _mean_abs(V_bc_hold_mid) if V_bc_hold_mid is not None else _mean_abs(V_bc_hold)
                oracle_in_mean_abs = _mean_abs(V_in_hold_mid) if V_in_hold_mid is not None else _mean_abs(V_in_hold)
                if not (_finite(oracle_bc_mean_abs) and _finite(oracle_in_mean_abs)):
                    _count_preflight("holdout_denom_nonfinite_count")
                    holdout_reject_reasons.add("holdout_denom_nonfinite")
                    oracle_bc_mean_abs = _finite_or_fail(oracle_bc_mean_abs, HOLDOUT_FAIL_VALUE)
                    oracle_in_mean_abs = _finite_or_fail(oracle_in_mean_abs, HOLDOUT_FAIL_VALUE)
                lap_denom = _laplacian_denom(oracle_in_mean_abs, oracle_bc_mean_abs)
                if not _finite(lap_denom):
                    _count_preflight("holdout_denom_nonfinite_count")
                    holdout_reject_reasons.add("holdout_denom_nonfinite")
                    lap_denom = float(HOLDOUT_FAIL_VALUE)
                holdout_stats = {
                    "holdout_total": holdout_total,
                    "holdout_boundary": holdout_boundary,
                    "holdout_interior": holdout_interior,
                    "holdout_resampled": holdout_resampled,
                    "oracle_bc_mean_abs": float(oracle_bc_mean_abs),
                    "oracle_in_mean_abs": float(oracle_in_mean_abs),
                    "lap_denom": float(lap_denom),
                }
                if holdout_partition_flags:
                    holdout_stats["holdout_partition_flags"] = list(holdout_partition_flags)

                V_ref_bc_train = None
                V_ref_in_train = None
                V_ref_bc_hold = None
                V_ref_in_hold = None
                V_ref_train = None
                V_ref_hold = None
                V_bc_train_corr = V_bc_train
                V_in_train_corr = V_in_train
                V_bc_hold_corr = V_bc_hold
                V_in_hold_corr = V_in_hold
                V_bc_hold_mid_corr = V_bc_hold_mid
                V_in_hold_mid_corr = V_in_hold_mid
                if use_ref:
                    V_ref_bc_train64 = compute_layered_reference_potential(
                        spec,
                        bc_train.to(dtype=torch.float64),
                        device=device,
                        dtype=torch.float64,
                    )
                    V_ref_in_train64 = compute_layered_reference_potential(
                        spec,
                        interior_train.to(dtype=torch.float64),
                        device=device,
                        dtype=torch.float64,
                    )
                    V_ref_bc_hold64 = compute_layered_reference_potential(
                        spec,
                        bc_hold.to(dtype=torch.float64),
                        device=device,
                        dtype=torch.float64,
                    )
                    V_ref_in_hold64 = compute_layered_reference_potential(
                        spec,
                        interior_hold.to(dtype=torch.float64),
                        device=device,
                        dtype=torch.float64,
                    )
                    V_ref_bc_train = V_ref_bc_train64.to(dtype=bc_train.dtype)
                    V_ref_in_train = V_ref_in_train64.to(dtype=interior_train.dtype)
                    V_ref_bc_hold = V_ref_bc_hold64.to(dtype=bc_hold.dtype)
                    V_ref_in_hold = V_ref_in_hold64.to(dtype=interior_hold.dtype)
                    V_ref_train = torch.cat([V_ref_bc_train, V_ref_in_train], dim=0)
                    V_ref_hold = torch.cat([V_ref_bc_hold, V_ref_in_hold], dim=0)
                    V_bc_train_corr = stable_subtract_reference(
                        V_bc_train,
                        V_ref_bc_train64,
                        out_dtype=bc_train.dtype,
                    )
                    V_in_train_corr = stable_subtract_reference(
                        V_in_train,
                        V_ref_in_train64,
                        out_dtype=interior_train.dtype,
                    )
                    V_bc_hold_corr = stable_subtract_reference(
                        V_bc_hold,
                        V_ref_bc_hold64,
                        out_dtype=bc_hold.dtype,
                    )
                    V_in_hold_corr = stable_subtract_reference(
                        V_in_hold,
                        V_ref_in_hold64,
                        out_dtype=interior_hold.dtype,
                    )
                    V_bc_hold_mid_corr = stable_subtract_reference(
                        V_bc_hold_mid,
                        V_ref_bc_hold64,
                        out_dtype=bc_hold.dtype,
                    )
                    V_in_hold_mid_corr = stable_subtract_reference(
                        V_in_hold_mid,
                        V_ref_in_hold64,
                        out_dtype=interior_hold.dtype,
                    )

                X_train = torch.cat([bc_train, interior_train], dim=0)
                V_train = torch.cat([V_bc_train_corr, V_in_train_corr], dim=0)
                is_boundary = torch.cat(
                    [torch.ones(bc_train.shape[0], device=device, dtype=torch.bool),
                     torch.zeros(interior_train.shape[0], device=device, dtype=torch.bool)],
                    dim=0,
                )
                interface_constraint_data = None
                if layered_enforce_interface_constraints and is_layered:
                    interface_constraint_data = _build_interface_constraint_data(
                        spec,
                        device=device,
                        dtype=X_train.dtype,
                        seed=seed_gen + 53,
                        domain_scale=domain_scale,
                        interface_delta=layered_interface_delta,
                        points_per_interface=layered_interface_constraint_points,
                        weight=layered_interface_constraint_weight,
                        use_ref=use_ref,
                    )
                X_hold = torch.cat([bc_hold, interior_hold], dim=0)
                V_hold = torch.cat([V_bc_hold, V_in_hold], dim=0)
                V_hold_mid = torch.cat([V_bc_hold_mid, V_in_hold_mid], dim=0)
                V_hold_corr = torch.cat([V_bc_hold_corr, V_in_hold_corr], dim=0)
                V_hold_mid_corr = torch.cat([V_bc_hold_mid_corr, V_in_hold_mid_corr], dim=0)
                is_boundary_hold = torch.cat(
                    [torch.ones(bc_hold.shape[0], device=device, dtype=torch.bool),
                     torch.zeros(interior_hold.shape[0], device=device, dtype=torch.bool)],
                    dim=0,
                )
                v_train_total_val = int(V_train.numel())
                v_train_nonfinite_val = 0
                v_train_has_nonfinite = False
                if preflight_counters is not None:
                    v_train_nonfinite_val = _nonfinite_count(V_train)
                    v_train_has_nonfinite = v_train_nonfinite_val > 0
                else:
                    v_train_has_nonfinite = not torch.isfinite(V_train).all().item()
                    if v_train_has_nonfinite:
                        v_train_nonfinite_val = _nonfinite_count(V_train)
                v_train_absmax = _tensor_absmax(V_train)
                v_train_frac_nonfinite = (
                    float(v_train_nonfinite_val) / max(1, v_train_total_val)
                    if v_train_total_val > 0
                    else 0.0
                )
                v_train_explosion = bool(v_train_absmax > 1e10 or v_train_frac_nonfinite > 0.0)
                v_train_nan_to_num_applied = False
                v_train_clamp_applied = False
                v_train_reference_subtracted = bool(use_ref)
                v_train_fit_absmax = v_train_absmax
                if interface_constraint_data is not None:
                    weight = float(interface_constraint_data.get("weight", 1.0))
                    ref_phi = interface_constraint_data.get("ref_phi")
                    ref_d = interface_constraint_data.get("ref_d")
                    v_constraints_absmax = max(_tensor_absmax(ref_phi), _tensor_absmax(ref_d))
                    if math.isfinite(v_constraints_absmax):
                        v_train_fit_absmax = max(v_train_fit_absmax, weight * v_constraints_absmax)

                fast_proxy_near_pts = torch.empty((0, 3), device=device, dtype=torch.float32)
                fast_proxy_far_pts = torch.empty((0, 3), device=device, dtype=torch.float32)
                fast_proxy_iface_up = torch.empty((0, 3), device=device, dtype=torch.float32)
                fast_proxy_iface_dn = torch.empty((0, 3), device=device, dtype=torch.float32)
                if fast_proxy_enabled and is_layered:
                    near_radius = max(1e-3, float(fast_proxy_near_radius) * float(domain_scale))
                    far_radius = max(near_radius * 2.0, float(fast_proxy_far_radius) * float(domain_scale))
                    n_far = max(8, int(fast_proxy_n_far))
                    fast_proxy_near_pts = _sample_sphere_points(
                        n_far,
                        near_radius,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 11,
                    )
                    fast_proxy_far_pts = _sample_sphere_points(
                        n_far,
                        far_radius,
                        device=device,
                        dtype=torch.float32,
                        seed=seed_gen + 13,
                    )
                    if fast_proxy_n_interface > 0:
                        fast_proxy_iface_up, fast_proxy_iface_dn = sample_layered_interface_pairs(
                            spec,
                            n_xy=int(fast_proxy_n_interface),
                            device=device,
                            dtype=torch.float32,
                            seed=seed_gen + 17,
                            delta=layered_interface_delta,
                            domain_scale=domain_scale,
                        )

                spec_embedding, _, _ = geo_encoder.encode(spec, device=device, dtype=torch.float32)
                spec_embedding = spec_embedding.to(device=device, dtype=torch.float32)
                spec_batch = [
                    SpecBatchItem(spec=spec, spec_meta=spec_meta, spec_embedding=spec_embedding, seed=None)
                    for _ in range(backoff.population_B)
                ]
                rollout = rollout_on_policy(
                    gfn.env,
                    gfn.policy,
                    spec_batch,
                    max_steps=max_steps,
                    temperature_schedule=None,
                )
                programs = [state.program for state in rollout.final_states or ()]
                if sanity_force_baseline and not use_param_sampler:
                    programs.extend(_baseline_programs_for_spec(spec))
                seeded = _seed_layered_templates(
                    spec,
                    max_steps=max_steps,
                    count=8,
                    allow_real_primitives=allow_real_primitives,
                )
                if seeded:
                    programs.extend(seeded)

                manual_elements: Dict[Program, List[ImageBasisElement]] = {}
                manual_meta: Dict[Program, Dict[str, object]] = {}
                if complex_boost_cfg.enabled and is_layered:
                    max_extra = max(0, backoff.population_B - len(programs))
                    if max_extra > 0:
                        manual_candidates = build_layered_complex_candidates(
                            spec,
                            device=device,
                            dtype=torch.float32,
                            seed=seed_gen + 19,
                            config=complex_boost_cfg,
                            max_terms=max_steps - 1,
                            domain_scale=domain_scale,
                            exclusion_radius=layered_exclusion_radius,
                            allow_real_primitives=allow_real_primitives,
                        )
                        if manual_candidates:
                            if len(manual_candidates) > max_extra:
                                manual_candidates = manual_candidates[:max_extra]
                            manual_elements = {c.program: c.elements for c in manual_candidates}
                            manual_meta = {c.program: c.meta for c in manual_candidates}
                            programs.extend([c.program for c in manual_candidates])

                _count_preflight("sampled_programs_total", len(programs))
                if not programs:
                    raise RuntimeError("No programs sampled; aborting generation.")
                program_lengths = [len(getattr(p, "nodes", []) or []) for p in programs]
                payload_map: Dict[Program, Any] = {}
                if use_param_sampler:
                    param_programs = [p for p in programs if p not in manual_elements]
                    if param_programs:
                        program_batch = gfn.param_sampler.build_program_batch(
                            param_programs,
                            device=device,
                            max_ast_len=gfn.flow_config.max_ast_len,
                            max_tokens=gfn.flow_config.max_tokens,
                        )
                        payload = gfn.param_sampler.sample(
                            program_batch,
                            spec,
                            spec_embedding,
                            seed=seed_gen,
                            device=device,
                            dtype=gfn.flow_dtype,
                            n_steps=gfn.flow_config.n_steps,
                            solver=gfn.flow_config.solver,
                            temperature=gfn.flow_config.temperature,
                            latent_clip=gfn.flow_config.latent_clip,
                            max_tokens=gfn.flow_config.max_tokens,
                            max_ast_len=gfn.flow_config.max_ast_len,
                        )
                        payload_map = {
                            program: payload.for_program(idx)
                            for idx, program in enumerate(param_programs)
                        }

                fast_scores: List[float] = []
                fast_metrics: List[Dict[str, Any]] = []
                n_terms_list: List[int] = []
                complex_counts: List[int] = []
                candidate_signatures: Dict[int, str] = {}
                gen_candidates_total = 0
                gen_complex_candidates = 0
                gen_dcim_candidates = 0
                hold_cache: Dict[int, torch.Tensor] = {} if cache_hold_matrices else {}
                for start in range(0, len(programs), score_microbatch):
                    end = min(len(programs), start + score_microbatch)
                    for idx in range(start, end):
                        program = programs[idx]
                        try:
                            manual = manual_elements.get(program)
                            if manual is not None:
                                elements = manual
                                meta = manual_meta.get(program, {})
                            else:
                                per_payload = payload_map.get(program)
                                if per_payload is not None:
                                    elements, _, meta = compile_program_to_basis(
                                        program,
                                        spec,
                                        device,
                                        param_payload=per_payload,
                                        strict=True,
                                    )
                                else:
                                    elements, _, meta = compile_program_to_basis(
                                        program,
                                        spec,
                                        device,
                                        strict=False,
                                    )
                        except Exception:
                            _count_preflight("compiled_failed")
                            raise
                        if not elements:
                            _count_preflight("compiled_empty_basis")
                            fast_scores.append(float("inf"))
                            fast_metrics.append({"empty": True, "complex_count": 0, "frac_complex": 0.0})
                            n_terms_list.append(0)
                            complex_counts.append(0)
                            continue
                        _count_preflight("compiled_ok")
                        complex_count = _count_complex_terms(elements, device=device)
                        n_terms = int(len(elements))
                        frac_complex = float(complex_count) / max(1, n_terms)
                        complex_counts.append(complex_count)
                        dcim_stats = _dcim_stats(elements)
                        dcim_terms = int(dcim_stats.get("dcim_terms", 0))
                        complex_present = bool(complex_count > 0 or dcim_terms > 0)
                        imag_max = _max_abs_imag_depth(elements, device=device) if preflight_enabled or dcim_diversity else 0.0
                        imag_bucket = log10_bucket(imag_max) if imag_max > 0.0 else "0"
                        candidate_signatures[idx] = _candidate_signature(dcim_stats, imag_bucket)
                        gen_candidates_total += 1
                        if complex_present:
                            gen_complex_candidates += 1
                            _count_preflight("complex_candidates")
                        if dcim_terms > 0:
                            gen_dcim_candidates += 1
                            _count_preflight("dcim_candidates")
                        if preflight_counters is not None:
                            _count_preflight("dcim_pole_terms", int(dcim_stats.get("dcim_poles", 0)))
                            _count_preflight("dcim_branch_terms", int(dcim_stats.get("dcim_branches", 0)))
                            _count_preflight("dcim_block_terms", int(dcim_stats.get("dcim_blocks", 0)))
                        if preflight_enabled:
                            _update_hist(dcim_pole_hist, dcim_stats.get("dcim_poles", 0))
                            _update_hist(dcim_block_hist, dcim_stats.get("dcim_blocks", 0))
                            if imag_max > 0.0 and math.isfinite(imag_max):
                                _update_hist(max_imag_hist, log10_bucket(imag_max))
                        if is_layered and not (complex_count >= 2 or frac_complex >= 0.25):
                            _count_preflight("complex_guard_failed")
                            fast_scores.append(float("inf"))
                            fast_metrics.append(
                                {
                                    "complex_guard_fail": True,
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue
                        if is_layered and layered_exclusion_radius > 0.0:
                            if _violates_interface_exclusion(
                                elements,
                                interface_planes,
                                layered_exclusion_radius,
                                device=device,
                            ):
                                _count_preflight("interface_exclusion_reject")
                                fast_scores.append(float("inf"))
                                fast_metrics.append(
                                    {
                                        "interface_exclusion_reject": True,
                                        "complex_count": int(complex_count),
                                        "frac_complex": float(frac_complex),
                                    }
                                )
                                n_terms_list.append(n_terms)
                                continue
                        try:
                            A_train = assemble_basis_matrix(elements, X_train)
                            A_hold = assemble_basis_matrix(elements, X_hold)
                        except Exception:
                            _count_preflight("assembled_failed")
                            raise
                        _count_preflight("assembled_ok")
                        assert_cuda_tensor(A_train, "A_train_fast")
                        assert_cuda_tensor(A_hold, "A_hold_fast")
                        V_train_fit = V_train
                        is_boundary_fit = is_boundary
                        if interface_constraint_data is not None:
                            A_train, V_train_fit, is_boundary_fit = _apply_interface_constraints(
                                A_train=A_train,
                                V_train=V_train_fit,
                                is_boundary=is_boundary_fit,
                                elements=elements,
                                constraint_data=interface_constraint_data,
                            )
                        if v_train_explosion and not v_train_snapshot_written:
                            payload = build_vtrain_explosion_snapshot(
                                spec,
                                X_train,
                                V_train_fit,
                                A_train,
                                layered_reference_enabled=use_ref,
                                reference_subtracted_for_fit=v_train_reference_subtracted,
                                nan_to_num_applied=v_train_nan_to_num_applied,
                                clamp_applied=v_train_clamp_applied,
                                seed=seed_gen,
                                gen=gen,
                                program_idx=idx,
                            )
                            if write_vtrain_explosion_snapshot(run_dir, payload):
                                v_train_snapshot_written = True
                        if cache_hold_matrices:
                            hold_cache[idx] = A_hold
                        a_train_nonfinite = 0
                        a_hold_nonfinite = 0
                        if preflight_full:
                            a_train_nonfinite = _nonfinite_count(A_train)
                            a_hold_nonfinite = _nonfinite_count(A_hold)
                            _count_preflight("a_train_nonfinite_count", a_train_nonfinite)
                            _count_preflight("a_train_total", int(A_train.numel()))
                            _count_preflight("a_hold_nonfinite_count", a_hold_nonfinite)
                            _count_preflight("a_hold_total", int(A_hold.numel()))
                            _count_preflight("v_train_nonfinite_count", v_train_nonfinite_val)
                            _count_preflight("v_train_total", v_train_total_val)
                        elif preflight_enabled:
                            _count_preflight("v_train_nonfinite_count", v_train_nonfinite_val)
                            _count_preflight("v_train_total", v_train_total_val)
                        else:
                            if not torch.isfinite(A_train).all().item():
                                a_train_nonfinite = 1
                            if not torch.isfinite(A_hold).all().item():
                                a_hold_nonfinite = 1
                        if (preflight_full and (a_train_nonfinite > 0 or a_hold_nonfinite > 0)) or v_train_has_nonfinite:
                            _count_preflight("solved_failed")
                            if v_train_has_nonfinite:
                                reason = "v_train_nonfinite"
                            elif a_train_nonfinite > 0:
                                reason = "a_train_nonfinite"
                            else:
                                reason = "a_hold_nonfinite"
                            _maybe_write_first_offender(
                                gen_idx=gen,
                                program_idx=idx,
                                program=program,
                                elements=elements,
                                A_train=A_train,
                                V_train=V_train_fit,
                                weights=None,
                                pred_hold=None,
                                reason=reason,
                            )
                            fast_scores.append(float("inf"))
                            fast_metrics.append(
                                {
                                    "nonfinite_matrix": True,
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue

                        cond_ratio = float("nan")
                        if preflight_enabled or fast_proxy_cond_weight > 0.0 or fast_proxy_cond_max > 0.0:
                            cond_ratio = condition_ratio(A_train)
                            if preflight_enabled and math.isfinite(cond_ratio):
                                _update_hist(cond_ratio_hist, log10_bucket(cond_ratio))
                            if fast_proxy_cond_max > 0.0 and (
                                not math.isfinite(cond_ratio) or cond_ratio > fast_proxy_cond_max
                            ):
                                _count_preflight("fast_proxy_condition_reject")
                                fast_scores.append(float("inf"))
                                fast_metrics.append(
                                    {
                                        "condition_reject": True,
                                        "cond_ratio": cond_ratio,
                                        "complex_count": int(complex_count),
                                        "frac_complex": float(frac_complex),
                                    }
                                )
                                n_terms_list.append(n_terms)
                                continue

                        weights = _fast_weights(
                            A_train,
                            V_train_fit,
                            reg=float(solver_cfg.get("reg_l1", 1e-3)),
                            normalize=bool(solver_cfg.get("fast_column_normalize", True)),
                            max_abs_b=v_train_fit_absmax,
                        )
                        weights_nonfinite = 0
                        if preflight_counters is not None:
                            _count_preflight("weights_total", int(weights.numel()))
                            if weights.numel() > 0:
                                weights_nonfinite = _nonfinite_count(weights)
                                _count_preflight("weights_nonfinite_count", weights_nonfinite)
                        if preflight_counters is not None:
                            if weights.numel() == 0 or weights_nonfinite > 0:
                                _count_preflight("solved_failed")
                                if weights.numel() == 0:
                                    _count_preflight("weights_empty")
                            else:
                                _count_preflight("solved_ok")
                        if weights.numel() == 0:
                            fast_scores.append(float("inf"))
                            fast_metrics.append(
                                {
                                    "weights_empty": True,
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue
                        if weights_nonfinite > 0 or not torch.isfinite(weights).all().item():
                            _maybe_write_first_offender(
                                gen_idx=gen,
                                program_idx=idx,
                                program=program,
                                elements=elements,
                                A_train=A_train,
                                V_train=V_train_fit,
                                weights=weights,
                                pred_hold=None,
                                reason="weights_nonfinite",
                            )
                            fast_scores.append(float("inf"))
                            fast_metrics.append(
                                {
                                    "weights_nonfinite": True,
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue
                        max_abs_weight = float(torch.max(torch.abs(weights)).item()) if weights.numel() > 0 else 0.0
                        if preflight_enabled and math.isfinite(max_abs_weight):
                            _update_hist(max_weight_hist, log10_bucket(max_abs_weight))
                        pred_hold_corr = A_hold.matmul(weights)
                        pred_hold = _add_reference(pred_hold_corr, V_ref_hold)
                        if preflight_counters is not None:
                            nonfinite = _nonfinite_count(pred_hold)
                            _count_preflight("nonfinite_pred_count", int(nonfinite))
                            _count_preflight("nonfinite_pred_total", int(pred_hold.numel()))
                            if nonfinite > 0:
                                _maybe_write_first_offender(
                                    gen_idx=gen,
                                    program_idx=idx,
                                    program=program,
                                    elements=elements,
                                    A_train=A_train,
                                    V_train=V_train_fit,
                                    weights=weights,
                                    pred_hold=pred_hold,
                                    reason="pred_nonfinite",
                                )
                                fast_scores.append(float("inf"))
                                fast_metrics.append(
                                    {
                                        "pred_nonfinite": True,
                                        "complex_count": int(complex_count),
                                        "frac_complex": float(frac_complex),
                                    }
                                )
                                n_terms_list.append(n_terms)
                                continue
                        elif not torch.isfinite(pred_hold).all().item():
                            _maybe_write_first_offender(
                                gen_idx=gen,
                                program_idx=idx,
                                program=program,
                                elements=elements,
                                A_train=A_train,
                                V_train=V_train_fit,
                                weights=weights,
                                pred_hold=pred_hold,
                                reason="pred_nonfinite",
                            )
                            fast_scores.append(float("inf"))
                            fast_metrics.append(
                                {
                                    "pred_nonfinite": True,
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue
                        bc_err = _mean_safe(torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold]))
                        in_err = _mean_safe(torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold]))
                        holdout_ok, holdout_vals, holdout_nonfinite = _sanitize_metric_block(
                            {"mean_bc": bc_err, "mean_in": in_err},
                            fail_value=HOLDOUT_FAIL_VALUE,
                        )
                        holdout_nonfinite_fields = holdout_nonfinite + holdout_partition_flags
                        if holdout_partition_flags:
                            holdout_ok = False
                        if not holdout_ok:
                            _count_preflight("holdout_nonfinite_candidate_count")
                            if "mean_in" in holdout_nonfinite:
                                _count_preflight("interior_metric_nonfinite_count")
                            reason = "holdout_nonfinite_fast"
                            if holdout_partition_flags:
                                reason = "holdout_partition_empty"
                            _log_holdout_nonfinite(
                                gen_idx=gen,
                                program_idx=idx,
                                program=program,
                                elements=elements,
                                A_train=A_train,
                                V_train=V_train_fit,
                                weights=weights,
                                pred_hold=pred_hold,
                                V_hold=V_hold,
                                V_hold_mid=V_hold_mid,
                                is_boundary_hold=is_boundary_hold,
                                lap=None,
                                denom_in=oracle_in_mean_abs,
                                denom_lap=lap_denom,
                                reason=reason,
                            )
                            fast_scores.append(float(HOLDOUT_FAIL_VALUE))
                            _count_preflight("fast_scored")
                            fast_metrics.append(
                                {
                                    "holdout_nonfinite": True,
                                    "holdout_nonfinite_fields": holdout_nonfinite_fields,
                                    "bc_mean_abs": holdout_vals["mean_bc"],
                                    "pde_mean_abs": holdout_vals["mean_in"],
                                    "complex_count": int(complex_count),
                                    "frac_complex": float(frac_complex),
                                }
                            )
                            n_terms_list.append(n_terms)
                            continue
                        fast_far_ratio = float("nan")
                        fast_iface_jump = float("nan")
                        fast_far_penalty = 0.0
                        fast_iface_penalty = 0.0
                        cond_penalty = 0.0
                        speed_proxy = _speed_proxy(
                            n_terms,
                            dcim_terms,
                            int(dcim_stats.get("dcim_blocks", 0)),
                            dcim_term_cost=fast_proxy_dcim_term_cost,
                            dcim_block_cost=fast_proxy_dcim_block_cost,
                        )
                        if fast_proxy_cond_weight > 0.0 and math.isfinite(cond_ratio):
                            cond_penalty = max(0.0, cond_ratio - fast_proxy_cond_target)
                        if fast_proxy_enabled and is_layered and (
                            fast_proxy_far_weight > 0.0
                            or fast_proxy_interface_weight > 0.0
                            or fast_proxy_fail_hard
                        ):
                            system = ImageSystem(elements, weights)

                            def _fast_eval(pts: torch.Tensor) -> torch.Tensor:
                                out = system.potential(pts)
                                if use_ref:
                                    out = out + compute_layered_reference_potential(
                                        spec,
                                        pts,
                                        device=pts.device,
                                        dtype=pts.dtype,
                                    )
                                return out

                            if fast_proxy_far_weight > 0.0 or fast_proxy_fail_hard:
                                fast_far_ratio = far_field_ratio(
                                    _fast_eval, fast_proxy_near_pts, fast_proxy_far_pts
                                )
                                if math.isfinite(fast_far_ratio):
                                    fast_far_penalty = max(0.0, fast_far_ratio - fast_proxy_far_target)
                                if fast_proxy_fail_hard and math.isfinite(fast_far_ratio) and (
                                    fast_far_ratio > fast_proxy_far_max
                                ):
                                    _count_preflight("fast_proxy_far_reject")
                                    fast_scores.append(float("inf"))
                                    fast_metrics.append(
                                        {
                                            "fast_proxy_far_reject": True,
                                            "fast_proxy_far_ratio": fast_far_ratio,
                                            "complex_count": int(complex_count),
                                            "frac_complex": float(frac_complex),
                                        }
                                    )
                                    n_terms_list.append(n_terms)
                                    continue
                            if fast_proxy_interface_weight > 0.0 or fast_proxy_fail_hard:
                                fast_iface_jump = interface_jump(
                                    _fast_eval, fast_proxy_iface_up, fast_proxy_iface_dn
                                )
                                iface_limit = fast_proxy_interface_max
                                if iface_limit <= 0.0:
                                    iface_limit = float(verify_plan.thresholds.get("bc_continuity", 0.0))
                                if math.isfinite(fast_iface_jump) and iface_limit > 0.0:
                                    fast_iface_penalty = max(0.0, fast_iface_jump - iface_limit)
                                if fast_proxy_fail_hard and fast_proxy_interface_max > 0.0 and math.isfinite(fast_iface_jump):
                                    if fast_iface_jump > fast_proxy_interface_max:
                                        _count_preflight("fast_proxy_interface_reject")
                                        fast_scores.append(float("inf"))
                                        fast_metrics.append(
                                            {
                                                "fast_proxy_interface_reject": True,
                                                "fast_proxy_interface_jump": fast_iface_jump,
                                                "complex_count": int(complex_count),
                                                "frac_complex": float(frac_complex),
                                            }
                                        )
                                        n_terms_list.append(n_terms)
                                        continue
                        comp = _complexity(program, elements)
                        score = w_bc * holdout_vals["mean_bc"] + w_pde * holdout_vals["mean_in"] + w_complexity * comp["n_terms"]
                        if fast_proxy_speed_weight > 0.0:
                            score = score + fast_proxy_speed_weight * speed_proxy
                        if fast_proxy_far_weight > 0.0:
                            score = score + fast_proxy_far_weight * fast_far_penalty
                        if fast_proxy_interface_weight > 0.0:
                            score = score + fast_proxy_interface_weight * fast_iface_penalty
                        if fast_proxy_cond_weight > 0.0:
                            score = score + fast_proxy_cond_weight * cond_penalty
                        if not _finite(score):
                            _count_preflight("holdout_nonfinite_candidate_count")
                            _log_holdout_nonfinite(
                                gen_idx=gen,
                                program_idx=idx,
                                program=program,
                                elements=elements,
                                A_train=A_train,
                                V_train=V_train_fit,
                                weights=weights,
                                pred_hold=pred_hold,
                                V_hold=V_hold,
                                V_hold_mid=V_hold_mid,
                                is_boundary_hold=is_boundary_hold,
                                lap=None,
                                denom_in=oracle_in_mean_abs,
                                denom_lap=lap_denom,
                                reason="score_nonfinite_fast",
                            )
                            score = float(HOLDOUT_FAIL_VALUE)
                        fast_scores.append(float(score))
                        _count_preflight("fast_scored")
                        n_terms_list.append(int(comp["n_terms"]))
                        fast_metrics.append(
                            {
                                "bc_mean_abs": holdout_vals["mean_bc"],
                                "pde_mean_abs": holdout_vals["mean_in"],
                                "n_terms": comp["n_terms"],
                                "program_hash": meta.get("program_hash"),
                                "complex_count": int(complex_count),
                                "frac_complex": float(frac_complex),
                                "manual_candidate": bool(meta.get("manual_candidate", False)),
                                "cond_ratio": cond_ratio,
                                "speed_proxy": speed_proxy,
                                "fast_proxy_far_ratio": fast_far_ratio,
                                "fast_proxy_interface_jump": fast_iface_jump,
                            }
                        )

                gen_fraction_complex = 0.0
                gen_fraction_dcim = 0.0
                if gen_candidates_total > 0:
                    gen_fraction_complex = float(gen_complex_candidates) / float(gen_candidates_total)
                    gen_fraction_dcim = float(gen_dcim_candidates) / float(gen_candidates_total)

                if diversity_guard and gen == 0 and programs:
                    # Small CPU-side histogram for control logic; avoids GPU-only complexity.
                    short_terms = sum(1 for n in n_terms_list if n <= 2)
                    short_len = sum(1 for n in program_lengths if n <= 3)
                    frac_terms = short_terms / max(1, len(n_terms_list))
                    frac_len = short_len / max(1, len(program_lengths))
                    if frac_terms > 0.80 or frac_len > 0.80:
                        def _hist(vals: Sequence[int]) -> Dict[int, int]:
                            out: Dict[int, int] = {}
                            for v in vals:
                                v = int(v)
                                out[v] = out.get(v, 0) + 1
                            return dict(sorted(out.items()))

                        msg = (
                            "Diversity guard failed: too many short programs/terms "
                            f"(n_terms<=2: {frac_terms:.2%}, len<=3: {frac_len:.2%}). "
                            f"program_len_hist={_hist(program_lengths)}, "
                            f"n_terms_hist={_hist(n_terms_list)}"
                        )
                        run_logger.error("%s", msg)
                        raise RuntimeError(msg)
                    if is_layered:
                        zero_complex = sum(1 for c in complex_counts if c == 0)
                        frac_zero = zero_complex / max(1, len(complex_counts))
                        if frac_zero > 0.80:
                            msg = (
                                "Complex-image guard failed: z_imag stayed zero; "
                                f"compiler mapping or point eval not active (zero_complex={frac_zero:.2%})."
                            )
                            run_logger.error("%s", msg)
                            raise RuntimeError(msg)

                top_fast_idx = _select_topk(fast_scores, topK_fast)
                proxy_metrics_by_idx: Dict[int, Dict[str, Any]] = {}
                if use_gate_proxies and is_layered and top_fast_idx:
                    proxy_n_xy = int(verify_plan.samples.get("B_boundary", 96))
                    proxy_n_dir = int(min(verify_plan.samples.get("C_far", 96), verify_plan.samples.get("C_near", 96)))
                    proxy_n_dir = max(16, proxy_n_dir)
                    proxy_n_interior = int(verify_plan.samples.get("A_interior", 128))
                    lap_linf_tol = float(verify_plan.thresholds.get("laplacian_linf", 5e-3))
                    lap_fd_h = float(verify_plan.thresholds.get("laplacian_fd_h", 2e-2))
                    lap_exclusion_radius = float(verify_plan.thresholds.get("laplacian_exclusion_radius", 5e-2))
                    lap_prefer_autograd = bool(verify_plan.thresholds.get("laplacian_prefer_autograd", 1.0))
                    lap_interface_band = float(verify_plan.thresholds.get("laplacian_interface_band", 0.0))
                    proxy_pts = build_proxy_stability_points(
                        bc_hold,
                        interior_hold,
                        int(verify_plan.samples.get("D_points", 128)),
                    )
                    for idx in list(top_fast_idx):
                        program = programs[idx]
                        manual = manual_elements.get(program)
                        if manual is not None:
                            elements = manual
                        else:
                            per_payload = payload_map.get(program)
                            if per_payload is not None:
                                elements, _, _ = compile_program_to_basis(
                                    program,
                                    spec,
                                    device,
                                    param_payload=per_payload,
                                    strict=True,
                                )
                            else:
                                elements, _, _ = compile_program_to_basis(
                                    program,
                                    spec,
                                    device,
                                    strict=False,
                                )
                        if not elements:
                            continue
                        A_train = assemble_basis_matrix(elements, X_train)
                        assert_cuda_tensor(A_train, "A_train_proxy")
                        V_train_fit = V_train
                        if interface_constraint_data is not None:
                            A_train, V_train_fit, _ = _apply_interface_constraints(
                                A_train=A_train,
                                V_train=V_train_fit,
                                is_boundary=is_boundary,
                                elements=elements,
                                constraint_data=interface_constraint_data,
                            )
                        weights = _fast_weights(
                            A_train,
                            V_train_fit,
                            reg=float(solver_cfg.get("reg_l1", 1e-3)),
                            max_abs_b=v_train_fit_absmax,
                        )
                        if weights.numel() == 0:
                            continue
                        system = ImageSystem(elements, weights)

                        def _proxy_eval(pts: torch.Tensor) -> torch.Tensor:
                            out = system.potential(pts)
                            if use_ref:
                                out = out + compute_layered_reference_potential(
                                    spec,
                                    pts,
                                    device=pts.device,
                                    dtype=pts.dtype,
                                )
                            return out

                        proxy_metrics: Dict[str, Any] = {}
                        if use_gateA_proxy:
                            proxy_metrics.update(
                                proxy_gateA(
                                    spec,
                                    _proxy_eval,
                                    n_interior=proxy_n_interior,
                                    exclusion_radius=lap_exclusion_radius,
                                    fd_h=lap_fd_h,
                                    prefer_autograd=lap_prefer_autograd,
                                    interface_band=lap_interface_band,
                                    device=device,
                                    dtype=torch.float32,
                                    seed=seed_gen + 1,
                                    linf_tol=lap_linf_tol,
                                )
                            )
                        proxy_metrics.update(
                            proxy_gateB(
                                spec,
                                _proxy_eval,
                                n_xy=proxy_n_xy,
                                delta=layered_interface_delta,
                                device=device,
                                dtype=torch.float32,
                                seed=seed_gen + 3,
                            )
                        )
                        proxy_metrics.update(
                            proxy_gateC(
                                _proxy_eval,
                                near_radii=(0.125, 0.5),
                                far_radii=(10.0, 20.0),
                                n_dir=proxy_n_dir,
                                device=device,
                                dtype=torch.float32,
                                seed=seed_gen + 5,
                            )
                        )
                        proxy_metrics.update(
                            proxy_gateD(
                                _proxy_eval,
                                proxy_pts,
                                delta=layered_stability_delta,
                                seed=seed_gen + 7,
                            )
                        )
                        proxy_metrics["proxy_fail_count"] = _proxy_fail_count(proxy_metrics, verify_plan.thresholds)
                        proxy_score, score_sanitized = _proxy_score_with_sanitized(
                            proxy_metrics,
                            a_weight=proxyA_weight,
                            a_cap=proxyA_cap,
                            a_transform=proxyA_transform,
                        )
                        proxy_metrics["proxy_score"] = proxy_score
                        if score_sanitized:
                            proxy_metrics["proxy_score_nonfinite_sanitized"] = True
                            _count_preflight("proxy_score_nonfinite_sanitized")
                        _count_preflight("proxy_computed_count")
                        proxy_metrics_by_idx[idx] = proxy_metrics
                        if idx < len(fast_metrics):
                            fast_metrics[idx].update(proxy_metrics)

                    if proxy_ranking_mode == "balanced":
                        top_fast_idx = sorted(
                            top_fast_idx,
                            key=lambda i: (
                                _proxy_fail_count_noA(proxy_metrics_by_idx.get(i, {}), verify_plan.thresholds),
                                _proxyA_effective_ratio(
                                    proxy_metrics_by_idx.get(i, {}),
                                    proxyA_cap,
                                    proxyA_transform,
                                ),
                                _proxy_score(
                                    proxy_metrics_by_idx.get(i, {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                fast_scores[i],
                            ),
                        )[: max(1, min(topK_fast, len(top_fast_idx)))]
                    else:
                        top_fast_idx = sorted(
                            top_fast_idx,
                            key=lambda i: (
                                _proxy_fail_count(proxy_metrics_by_idx.get(i, {}), verify_plan.thresholds),
                                _proxy_score(
                                    proxy_metrics_by_idx.get(i, {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                fast_scores[i],
                            ),
                        )[: max(1, min(topK_fast, len(top_fast_idx)))]
                if dcim_diversity and is_layered and top_fast_idx:
                    top_fast_idx = _diversity_select(top_fast_idx, candidate_signatures, topK_fast)
                if cache_hold_matrices and hold_cache:
                    keep = set(top_fast_idx)
                    hold_cache = {idx: mat for idx, mat in hold_cache.items() if idx in keep}

                fitted_candidates: List[Dict[str, Any]] = []
                if use_dcim_block and is_layered and dcim_block_max > 0:
                    try:
                        if spec_hash in dcim_block_cache:
                            dcim_elements = dcim_block_cache[spec_hash]
                        else:
                            dcim_elements = _build_dcim_block_elements(
                                spec,
                                device=device,
                                max_blocks=dcim_block_max,
                                cache_path=dcim_cache_path,
                            )
                            dcim_block_cache[spec_hash] = dcim_elements
                    except Exception as exc:
                        raise RuntimeError(f"DCIM block baseline failed: {exc}") from exc

                    if dcim_elements:
                        if preflight_enabled:
                            dcim_stats = _dcim_stats(dcim_elements)
                            _update_hist(dcim_pole_hist, dcim_stats.get("dcim_poles", 0))
                            _update_hist(dcim_block_hist, dcim_stats.get("dcim_blocks", 0))
                            imag_max = _max_abs_imag_depth(dcim_elements, device=device)
                            if imag_max > 0.0 and math.isfinite(imag_max):
                                _update_hist(max_imag_hist, log10_bucket(imag_max))
                        dcim_nodes = [
                            AddPrimitiveBlock(
                                family_name="dcim_block_baseline",
                                conductor_id=0,
                                motif_id=idx,
                            )
                            for idx in range(len(dcim_elements))
                        ]
                        dcim_nodes.append(StopProgram())
                        dcim_program = Program(nodes=tuple(dcim_nodes))

                        assert_cuda_tensor(X_train, "X_train")
                        A_train_dcim = assemble_basis_matrix(dcim_elements, X_train)
                        assert_cuda_tensor(A_train_dcim, "A_train_dcim")
                        V_train_fit = V_train
                        is_boundary_fit = is_boundary
                        if interface_constraint_data is not None:
                            A_train_dcim, V_train_fit, is_boundary_fit = _apply_interface_constraints(
                                A_train=A_train_dcim,
                                V_train=V_train_fit,
                                is_boundary=is_boundary_fit,
                                elements=dcim_elements,
                                constraint_data=interface_constraint_data,
                            )
                        A_train_scaled, col_norms = _scale_columns(A_train_dcim)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        weights_scaled, _ = solve_sparse(
                            A_train_scaled,
                            X_train,
                            V_train_fit,
                            is_boundary_fit,
                            _NullLogger(),
                            reg_l1=float(solver_cfg.get("reg_l1", 1e-3)),
                            solver=solver_mode,
                            max_iter=int(solver_cfg.get("max_iters", 64)),
                            tol=float(solver_cfg.get("tol", 1e-6)),
                            lambda_group=float(solver_cfg.get("lambda_group", 0.0)),
                            normalize_columns=False,
                        )
                        weights = weights_scaled / col_norms
                        if weights.numel() > 0:
                            assert_cuda_tensor(weights, "weights_dcim")
                            weights_ok, reason = _validate_weights(weights)
                            if not weights_ok:
                                _reject_invalid_weights(
                                    gen_idx=gen,
                                    program_idx=-1,
                                    program=dcim_program,
                                    elements=dcim_elements,
                                    A_train=A_train_dcim,
                                    V_train=V_train_fit,
                                    weights=weights,
                                    pred_hold=None,
                                    reason=reason,
                                )
                                weights = weights[:0]
                        if weights.numel() > 0:
                            end.record()
                            torch.cuda.synchronize()
                            t_solve_ms = float(start.elapsed_time(end))

                            system = ImageSystem(dcim_elements, weights)

                            A_hold_dcim = assemble_basis_matrix(dcim_elements, X_hold)
                            assert_cuda_tensor(A_hold_dcim, "A_hold_dcim")
                            pred_hold_corr = A_hold_dcim.matmul(weights)
                            pred_hold = _add_reference(pred_hold_corr, V_ref_hold)
                            bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                            in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                            mean_bc = _mean_safe(bc_err)
                            mean_in = _mean_safe(in_err)
                            max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                            max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                            mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            mean_bc_mid = _mean_safe(mid_bc_err)
                            mean_in_mid = _mean_safe(mid_in_err)
                            rel_bc = mean_bc_mid / max(oracle_bc_mean_abs, 1e-12)
                            rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                            def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                                assert_cuda_tensor(pts, "candidate_eval_points")
                                out = system.potential(pts)
                                if use_ref:
                                    out = out + compute_layered_reference_potential(
                                        spec,
                                        pts,
                                        device=pts.device,
                                        dtype=pts.dtype,
                                    )
                                assert_cuda_tensor(out, "candidate_eval_out")
                                return out

                            lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                            lap_abs = torch.abs(lap)
                            lap_mean = _mean_safe(lap_abs)
                            lap_max = float(torch.max(lap_abs).item()) if lap_abs.numel() else 0.0
                            rel_lap = lap_mean / lap_denom

                            eval_ms = _timed_cuda(_eval_fn, interior_hold, warmup=1, repeat=3)

                            perturbed = _perturb_elements(dcim_elements, stability_sigma, device=device)
                            stability_ratio = float("nan")
                            if perturbed is not None:
                                A_hold_pert = assemble_basis_matrix(perturbed, X_hold)
                                assert_cuda_tensor(A_hold_pert, "A_hold_dcim_pert")
                                pert_pred_corr = A_hold_pert.matmul(weights)
                                pert_pred = _add_reference(pert_pred_corr, V_ref_hold)
                                base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                                pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                                pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                                pert_err = w_bc * _mean_safe(pert_bc) + w_pde * _mean_safe(pert_in)
                                stability_ratio = float(pert_err / max(base_err, 1e-8))
                            if not math.isfinite(stability_ratio):
                                stability_ratio = 1.0

                            holdout_ok, holdout_vals, holdout_nonfinite = _sanitize_metric_block(
                                {
                                    "mean_bc": mean_bc,
                                    "mean_in": mean_in,
                                    "max_bc": max_bc,
                                    "max_in": max_in,
                                    "mean_bc_mid": mean_bc_mid,
                                    "mean_in_mid": mean_in_mid,
                                    "rel_bc": rel_bc,
                                    "rel_in": rel_in,
                                },
                                fail_value=HOLDOUT_FAIL_VALUE,
                            )
                            lap_ok, lap_vals, lap_nonfinite = _sanitize_metric_block(
                                {"lap_mean": lap_mean, "lap_max": lap_max, "rel_lap": rel_lap},
                                fail_value=HOLDOUT_FAIL_VALUE,
                            )
                            holdout_nonfinite_fields = holdout_nonfinite + lap_nonfinite + holdout_partition_flags
                            holdout_nonfinite_flag = bool(holdout_nonfinite_fields)
                            if holdout_nonfinite_flag:
                                _count_preflight("holdout_nonfinite_candidate_count")
                                _count_preflight("dcim_baseline_nonfinite_count")
                                if {"mean_in", "mean_in_mid", "rel_in", "max_in"} & set(holdout_nonfinite):
                                    _count_preflight("interior_metric_nonfinite_count")
                                if lap_nonfinite:
                                    _count_preflight("lap_metric_nonfinite_count")
                                reason = "holdout_nonfinite_dcim"
                                if holdout_partition_flags:
                                    reason = "holdout_partition_empty"
                                _log_holdout_nonfinite(
                                    gen_idx=gen,
                                    program_idx=-1,
                                    program=dcim_program,
                                    elements=dcim_elements,
                                    A_train=A_train_dcim,
                                    V_train=V_train_fit,
                                    weights=weights,
                                    pred_hold=pred_hold,
                                    V_hold=V_hold,
                                    V_hold_mid=V_hold_mid,
                                    is_boundary_hold=is_boundary_hold,
                                    lap=lap,
                                    denom_in=oracle_in_mean_abs,
                                    denom_lap=lap_denom,
                                    reason=reason,
                                )

                            mean_bc = holdout_vals["mean_bc"]
                            mean_in = holdout_vals["mean_in"]
                            max_bc = holdout_vals["max_bc"]
                            max_in = holdout_vals["max_in"]
                            mean_bc_mid = holdout_vals["mean_bc_mid"]
                            mean_in_mid = holdout_vals["mean_in_mid"]
                            rel_bc = holdout_vals["rel_bc"]
                            rel_in = holdout_vals["rel_in"]
                            lap_mean = lap_vals["lap_mean"]
                            lap_max = lap_vals["lap_max"]
                            rel_lap = lap_vals["rel_lap"]

                            eval_time_us = float(eval_ms * 1000.0) if _finite(eval_ms) else float(HOLDOUT_FAIL_VALUE)
                            solve_time_us = float(t_solve_ms * 1000.0) if _finite(t_solve_ms) else float(HOLDOUT_FAIL_VALUE)
                            if _finite(eval_ms) and _finite(t_solve_ms):
                                total_time_us = float((eval_ms + t_solve_ms) * 1000.0)
                            else:
                                total_time_us = float(HOLDOUT_FAIL_VALUE)

                            comp = _complexity(dcim_program, dcim_elements)
                            complex_count = _count_complex_terms(dcim_elements, device=device)
                            metrics = {
                                "max_bc_err_holdout": max_bc,
                                "mean_bc_err_holdout": mean_bc,
                                "max_pde_err_holdout": lap_max,
                                "mean_pde_err_holdout": lap_mean,
                                "mid_bc_mean_abs": mean_bc_mid,
                                "mid_pde_mean_abs": mean_in_mid,
                                "rel_bc_err_holdout": float(rel_bc),
                                "rel_pde_err_holdout": float(rel_in),
                                "lap_mean_abs_holdout": float(lap_mean),
                                "rel_lap_holdout": float(rel_lap),
                                "oracle_bc_mean_abs_holdout": float(oracle_bc_mean_abs),
                                "oracle_in_mean_abs_holdout": float(oracle_in_mean_abs),
                                "stability_ratio": stability_ratio,
                                "eval_time_us": eval_time_us,
                                "solve_time_us": solve_time_us,
                                "total_time_us": total_time_us,
                                "complexity_terms": comp["n_terms"],
                                "complexity_nodes": comp["n_nodes"],
                                "n_terms": comp["n_terms"],
                                "complex_count": int(complex_count),
                                "is_dcim_block_baseline": True,
                                "dcim_block_weight": float(dcim_block_weight),
                                "candidate_name": "dcim_block_baseline",
                            }
                            if holdout_nonfinite_flag:
                                metrics["holdout_nonfinite"] = True
                                metrics["holdout_nonfinite_fields"] = holdout_nonfinite_fields
                            if use_gate_proxies and is_layered:
                                proxy_pts = build_proxy_stability_points(
                                    bc_hold,
                                    interior_hold,
                                    int(verify_plan.samples.get("D_points", 128)),
                                )
                                proxy_metrics = {}
                                if use_gateA_proxy:
                                    proxy_metrics.update(
                                        proxy_gateA(
                                            spec,
                                            _eval_fn,
                                            n_interior=int(verify_plan.samples.get("A_interior", 128)),
                                            exclusion_radius=float(
                                                verify_plan.thresholds.get("laplacian_exclusion_radius", 5e-2)
                                            ),
                                            fd_h=float(verify_plan.thresholds.get("laplacian_fd_h", 2e-2)),
                                            prefer_autograd=bool(
                                                verify_plan.thresholds.get("laplacian_prefer_autograd", 1.0)
                                            ),
                                            interface_band=float(
                                                verify_plan.thresholds.get("laplacian_interface_band", 0.0)
                                            ),
                                            device=device,
                                            dtype=torch.float32,
                                            seed=seed_gen + 1,
                                            linf_tol=float(verify_plan.thresholds.get("laplacian_linf", 5e-3)),
                                        )
                                    )
                                proxy_metrics.update(
                                    proxy_gateB(
                                        spec,
                                        _eval_fn,
                                        n_xy=int(verify_plan.samples.get("B_boundary", 96)),
                                        delta=layered_interface_delta,
                                        device=device,
                                        dtype=torch.float32,
                                        seed=seed_gen + 3,
                                    )
                                )
                                proxy_metrics.update(
                                    proxy_gateC(
                                        _eval_fn,
                                        near_radii=(0.125, 0.5),
                                        far_radii=(10.0, 20.0),
                                        n_dir=int(min(verify_plan.samples.get("C_far", 96), verify_plan.samples.get("C_near", 96))),
                                        device=device,
                                        dtype=torch.float32,
                                        seed=seed_gen + 5,
                                    )
                                )
                                proxy_metrics.update(
                                    proxy_gateD(
                                        _eval_fn,
                                        proxy_pts,
                                        delta=layered_stability_delta,
                                        seed=seed_gen + 7,
                                    )
                                )
                                proxy_metrics["proxy_fail_count"] = _proxy_fail_count(proxy_metrics, verify_plan.thresholds)
                                proxy_score, score_sanitized = _proxy_score_with_sanitized(
                                    proxy_metrics,
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                )
                                proxy_metrics["proxy_score"] = proxy_score
                                if score_sanitized:
                                    proxy_metrics["proxy_score_nonfinite_sanitized"] = True
                                    _count_preflight("proxy_score_nonfinite_sanitized")
                                _count_preflight("proxy_computed_count")
                                metrics.update(proxy_metrics)
                            score_mid = (
                                w_bc * rel_bc
                                + w_pde * rel_lap
                                + w_complexity * comp["n_terms"]
                                + w_latency * (metrics["eval_time_us"] / 1e6)
                                + w_stability * stability_ratio
                            )
                            if not _finite(score_mid):
                                _count_preflight("holdout_nonfinite_candidate_count")
                                _count_preflight("dcim_baseline_nonfinite_count")
                                _log_holdout_nonfinite(
                                    gen_idx=gen,
                                    program_idx=-1,
                                    program=dcim_program,
                                    elements=dcim_elements,
                                    A_train=A_train_dcim,
                                    V_train=V_train_fit,
                                    weights=weights,
                                    pred_hold=pred_hold,
                                    V_hold=V_hold,
                                    V_hold_mid=V_hold_mid,
                                    is_boundary_hold=is_boundary_hold,
                                    lap=lap,
                                    denom_in=oracle_in_mean_abs,
                                    denom_lap=lap_denom,
                                    reason="score_nonfinite_dcim",
                                )
                                metrics["holdout_nonfinite"] = True
                                score_mid = float(HOLDOUT_FAIL_VALUE)
                            if not metrics.get("holdout_nonfinite") and math.isfinite(float(rel_bc)) and (
                                best_dcim_rel_bc is None or float(rel_bc) < best_dcim_rel_bc
                            ):
                                best_dcim_rel_bc = float(rel_bc)
                            if not metrics.get("holdout_nonfinite") and math.isfinite(float(rel_lap)) and (
                                best_dcim_rel_lap is None or float(rel_lap) < best_dcim_rel_lap
                            ):
                                best_dcim_rel_lap = float(rel_lap)
                            if not metrics.get("holdout_nonfinite") and math.isfinite(float(score_mid)) and (
                                best_dcim_score is None or float(score_mid) < best_dcim_score
                            ):
                                best_dcim_score = float(score_mid)
                            _count_preflight("mid_scored")
                            fitted_candidates.append(
                                {
                                    "program": dcim_program,
                                    "elements": dcim_elements,
                                    "weights": weights,
                                    "metrics": metrics,
                                    "score": float(score_mid),
                                }
                            )
                for rank, idx in enumerate(top_fast_idx):
                    program = programs[idx]
                    if is_layered:
                        n_terms = max(1, int(n_terms_list[idx])) if idx < len(n_terms_list) else 1
                        complex_count = int(complex_counts[idx]) if idx < len(complex_counts) else 0
                        if not (complex_count >= 2 or (complex_count / n_terms) >= 0.25):
                            continue
                    manual = manual_elements.get(program)
                    if manual is not None:
                        elements = manual
                        group_ids = None
                        meta = manual_meta.get(program, {})
                    else:
                        per_payload = payload_map.get(program)
                        if per_payload is not None:
                            elements, group_ids, meta = compile_program_to_basis(
                                program,
                                spec,
                                device,
                                param_payload=per_payload,
                                strict=True,
                            )
                        else:
                            elements, group_ids, meta = compile_program_to_basis(
                                program,
                                spec,
                                device,
                                strict=False,
                            )
                    if not elements:
                        continue

                    assert_cuda_tensor(X_train, "X_train")
                    A_train = assemble_basis_matrix(elements, X_train)
                    assert_cuda_tensor(A_train, "A_train_fit")
                    V_train_fit = V_train
                    is_boundary_fit = is_boundary
                    if interface_constraint_data is not None:
                        A_train, V_train_fit, is_boundary_fit = _apply_interface_constraints(
                            A_train=A_train,
                            V_train=V_train_fit,
                            is_boundary=is_boundary_fit,
                            elements=elements,
                            constraint_data=interface_constraint_data,
                        )
                    A_train_scaled, col_norms = _scale_columns(A_train)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    weights_scaled, _ = solve_sparse(
                        A_train_scaled,
                        X_train,
                        V_train_fit,
                        is_boundary_fit,
                        _NullLogger(),
                        reg_l1=float(solver_cfg.get("reg_l1", 1e-3)),
                        solver=solver_mode,
                        max_iter=int(solver_cfg.get("max_iters", 64)),
                        tol=float(solver_cfg.get("tol", 1e-6)),
                        lambda_group=float(solver_cfg.get("lambda_group", 0.0)),
                        normalize_columns=False,
                    )
                    weights = weights_scaled / col_norms
                    if weights.numel() == 0:
                        continue
                    assert_cuda_tensor(weights, "weights")
                    weights_ok, reason = _validate_weights(weights)
                    if not weights_ok:
                        _reject_invalid_weights(
                            gen_idx=gen,
                            program_idx=idx,
                            program=program,
                            elements=elements,
                            A_train=A_train,
                            V_train=V_train_fit,
                            weights=weights,
                            pred_hold=None,
                            reason=reason,
                        )
                        continue
                    end.record()
                    torch.cuda.synchronize()
                    t_solve_ms = float(start.elapsed_time(end))
                    system = ImageSystem(elements, weights)

                    if cache_hold_matrices and idx in hold_cache:
                        A_hold = hold_cache.pop(idx)
                    else:
                        A_hold = assemble_basis_matrix(elements, X_hold)
                    assert_cuda_tensor(A_hold, "A_hold_fit")
                    pred_hold_corr = A_hold.matmul(weights)
                    pred_hold = _add_reference(pred_hold_corr, V_ref_hold)
                    bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                    in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                    mean_bc = _mean_safe(bc_err)
                    mean_in = _mean_safe(in_err)
                    max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                    max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                    mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                    mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                    mean_bc_mid = _mean_safe(mid_bc_err)
                    mean_in_mid = _mean_safe(mid_in_err)
                    rel_bc = mean_bc_mid / max(oracle_bc_mean_abs, 1e-12)
                    rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                    def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                        assert_cuda_tensor(pts, "candidate_eval_points")
                        out = system.potential(pts)
                        if use_ref:
                            out = out + compute_layered_reference_potential(
                                spec,
                                pts,
                                device=pts.device,
                                dtype=pts.dtype,
                            )
                        assert_cuda_tensor(out, "candidate_eval_out")
                        return out

                    lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                    lap_abs = torch.abs(lap)
                    lap_mean = _mean_safe(lap_abs)
                    lap_max = float(torch.max(lap_abs).item()) if lap_abs.numel() else 0.0
                    rel_lap = lap_mean / lap_denom

                    eval_ms = _timed_cuda(_eval_fn, interior_hold, warmup=1, repeat=3)

                    perturbed = _perturb_elements(elements, stability_sigma, device=device)
                    stability_ratio = float("nan")
                    if perturbed is not None:
                        A_hold_pert = assemble_basis_matrix(perturbed, X_hold)
                        assert_cuda_tensor(A_hold_pert, "A_hold_pert")
                        pert_pred_corr = A_hold_pert.matmul(weights)
                        pert_pred = _add_reference(pert_pred_corr, V_ref_hold)
                        base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                        pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                        pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                        pert_err = w_bc * _mean_safe(pert_bc) + w_pde * _mean_safe(pert_in)
                        stability_ratio = float(pert_err / max(base_err, 1e-8))
                    if not math.isfinite(stability_ratio):
                        stability_ratio = 1.0

                    holdout_ok, holdout_vals, holdout_nonfinite = _sanitize_metric_block(
                        {
                            "mean_bc": mean_bc,
                            "mean_in": mean_in,
                            "max_bc": max_bc,
                            "max_in": max_in,
                            "mean_bc_mid": mean_bc_mid,
                            "mean_in_mid": mean_in_mid,
                            "rel_bc": rel_bc,
                            "rel_in": rel_in,
                        },
                        fail_value=HOLDOUT_FAIL_VALUE,
                    )
                    lap_ok, lap_vals, lap_nonfinite = _sanitize_metric_block(
                        {"lap_mean": lap_mean, "lap_max": lap_max, "rel_lap": rel_lap},
                        fail_value=HOLDOUT_FAIL_VALUE,
                    )
                    holdout_nonfinite_fields = holdout_nonfinite + lap_nonfinite + holdout_partition_flags
                    holdout_nonfinite_flag = bool(holdout_nonfinite_fields)
                    if holdout_nonfinite_flag:
                        _count_preflight("holdout_nonfinite_candidate_count")
                        if {"mean_in", "mean_in_mid", "rel_in", "max_in"} & set(holdout_nonfinite):
                            _count_preflight("interior_metric_nonfinite_count")
                        if lap_nonfinite:
                            _count_preflight("lap_metric_nonfinite_count")
                        reason = "holdout_nonfinite_mid"
                        if holdout_partition_flags:
                            reason = "holdout_partition_empty"
                        _log_holdout_nonfinite(
                            gen_idx=gen,
                            program_idx=idx,
                            program=program,
                            elements=elements,
                            A_train=A_train,
                            V_train=V_train_fit,
                            weights=weights,
                            pred_hold=pred_hold,
                            V_hold=V_hold,
                            V_hold_mid=V_hold_mid,
                            is_boundary_hold=is_boundary_hold,
                            lap=lap,
                            denom_in=oracle_in_mean_abs,
                            denom_lap=lap_denom,
                            reason=reason,
                        )

                    mean_bc = holdout_vals["mean_bc"]
                    mean_in = holdout_vals["mean_in"]
                    max_bc = holdout_vals["max_bc"]
                    max_in = holdout_vals["max_in"]
                    mean_bc_mid = holdout_vals["mean_bc_mid"]
                    mean_in_mid = holdout_vals["mean_in_mid"]
                    rel_bc = holdout_vals["rel_bc"]
                    rel_in = holdout_vals["rel_in"]
                    lap_mean = lap_vals["lap_mean"]
                    lap_max = lap_vals["lap_max"]
                    rel_lap = lap_vals["rel_lap"]

                    eval_time_us = float(eval_ms * 1000.0) if _finite(eval_ms) else float(HOLDOUT_FAIL_VALUE)
                    solve_time_us = float(t_solve_ms * 1000.0) if _finite(t_solve_ms) else float(HOLDOUT_FAIL_VALUE)
                    if _finite(eval_ms) and _finite(t_solve_ms):
                        total_time_us = float((eval_ms + t_solve_ms) * 1000.0)
                    else:
                        total_time_us = float(HOLDOUT_FAIL_VALUE)

                    comp = _complexity(program, elements)
                    complex_count = _count_complex_terms(elements, device=device)
                    metrics = {
                        "max_bc_err_holdout": max_bc,
                        "mean_bc_err_holdout": mean_bc,
                        "max_pde_err_holdout": lap_max,
                        "mean_pde_err_holdout": lap_mean,
                        "mid_bc_mean_abs": mean_bc_mid,
                        "mid_pde_mean_abs": mean_in_mid,
                        "rel_bc_err_holdout": float(rel_bc),
                        "rel_pde_err_holdout": float(rel_in),
                        "lap_mean_abs_holdout": float(lap_mean),
                        "rel_lap_holdout": float(rel_lap),
                        "oracle_bc_mean_abs_holdout": float(oracle_bc_mean_abs),
                        "oracle_in_mean_abs_holdout": float(oracle_in_mean_abs),
                        "stability_ratio": stability_ratio,
                        "eval_time_us": eval_time_us,
                        "solve_time_us": solve_time_us,
                        "total_time_us": total_time_us,
                        "complexity_terms": comp["n_terms"],
                        "complexity_nodes": comp["n_nodes"],
                        "n_terms": comp["n_terms"],
                        "complex_count": int(complex_count),
                    }
                    if holdout_nonfinite_flag:
                        metrics["holdout_nonfinite"] = True
                        metrics["holdout_nonfinite_fields"] = holdout_nonfinite_fields
                    if use_gate_proxies and is_layered:
                        proxy_metrics = proxy_metrics_by_idx.get(idx)
                        if proxy_metrics:
                            metrics.update(proxy_metrics)
                    score_mid = (
                        w_bc * rel_bc
                        + w_pde * rel_lap
                        + w_complexity * comp["n_terms"]
                        + w_latency * (metrics["eval_time_us"] / 1e6)
                        + w_stability * stability_ratio
                    )
                    if not _finite(score_mid):
                        _count_preflight("holdout_nonfinite_candidate_count")
                        _log_holdout_nonfinite(
                            gen_idx=gen,
                            program_idx=idx,
                            program=program,
                            elements=elements,
                            A_train=A_train,
                            V_train=V_train_fit,
                            weights=weights,
                            pred_hold=pred_hold,
                            V_hold=V_hold,
                            V_hold_mid=V_hold_mid,
                            is_boundary_hold=is_boundary_hold,
                            lap=lap,
                            denom_in=oracle_in_mean_abs,
                            denom_lap=lap_denom,
                            reason="score_nonfinite_mid",
                        )
                        metrics["holdout_nonfinite"] = True
                        score_mid = float(HOLDOUT_FAIL_VALUE)
                    _count_preflight("mid_scored")
                    fitted_candidates.append(
                        {
                            "program": program,
                            "elements": elements,
                            "weights": weights,
                            "metrics": metrics,
                            "score": float(score_mid),
                        }
                    )

                if use_gate_proxies and is_layered:
                    if proxy_ranking_mode == "balanced":
                        fitted_candidates.sort(
                            key=lambda x: (
                                _proxy_fail_count_noA(x.get("metrics", {}), verify_plan.thresholds),
                                _proxyA_effective_ratio(
                                    x.get("metrics", {}),
                                    proxyA_cap,
                                    proxyA_transform,
                                ),
                                _proxy_score(
                                    x.get("metrics", {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                x["score"],
                            )
                        )
                    else:
                        fitted_candidates.sort(
                            key=lambda x: (
                                _proxy_fail_count(x.get("metrics", {}), verify_plan.thresholds),
                                _proxy_score(
                                    x.get("metrics", {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                x["score"],
                            )
                        )
                else:
                    fitted_candidates.sort(key=lambda x: x["score"])
                top_mid = fitted_candidates[: max(1, min(topk_mid, len(fitted_candidates)))]

                top_final = top_mid
                if use_hi_oracle and top_mid:
                    hi_cfg = oracle_cfg.get("hi", {}) if isinstance(oracle_cfg.get("hi", {}), dict) else {}
                    hi_fidelity = _oracle_for_name(str(hi_cfg.get("name", "hi")))
                    hi_dtype = _resolve_dtype(str(hi_cfg.get("dtype", "fp32")))
                    oracle_hi = _build_oracle_backend(hi_fidelity, spec)
                    V_bc_hold_hi, _ = _oracle_eval(oracle_hi, spec, bc_hold, dtype=hi_dtype)
                    V_in_hold_hi, _ = _oracle_eval(oracle_hi, spec, interior_hold, dtype=hi_dtype)
                    V_bc_hold_hi = V_bc_hold_hi.to(dtype=bc_hold.dtype)
                    V_in_hold_hi = V_in_hold_hi.to(dtype=interior_hold.dtype)
                    V_hold_hi = torch.cat([V_bc_hold_hi, V_in_hold_hi], dim=0)
                    for cand in top_mid:
                        A_hold_hi = assemble_basis_matrix(cand["elements"], X_hold)
                        assert_cuda_tensor(A_hold_hi, "A_hold_hi")
                        pred_hi_corr = A_hold_hi.matmul(cand["weights"])
                        pred_hi = _add_reference(pred_hi_corr, V_ref_hold)
                        bc_err_hi = _mean_safe(torch.abs(pred_hi[is_boundary_hold] - V_hold_hi[is_boundary_hold]))
                        in_err_hi = _mean_safe(torch.abs(pred_hi[~is_boundary_hold] - V_hold_hi[~is_boundary_hold]))
                        cand["score_hi"] = float(w_bc * bc_err_hi + w_pde * in_err_hi)
                        cand["metrics"]["hi_bc_mean_abs"] = float(bc_err_hi)
                        cand["metrics"]["hi_pde_mean_abs"] = float(in_err_hi)
                    top_final = sorted(top_mid, key=lambda x: x.get("score_hi", x["score"]))[
                        : max(1, min(topk_final, len(top_mid)))
                    ]
                else:
                    top_final = top_mid[: max(1, min(topk_final, len(top_mid)))]

                if use_gate_proxies and is_layered and top_final:
                    if proxy_ranking_mode == "balanced":
                        top_final = sorted(
                            top_final,
                            key=lambda x: (
                                _proxy_fail_count_noA(x.get("metrics", {}), verify_plan.thresholds),
                                _proxyA_effective_ratio(
                                    x.get("metrics", {}),
                                    proxyA_cap,
                                    proxyA_transform,
                                ),
                                _proxy_score(
                                    x.get("metrics", {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                x.get("score_hi", x["score"]),
                            ),
                        )
                    else:
                        top_final = sorted(
                            top_final,
                            key=lambda x: (
                                _proxy_fail_count(x.get("metrics", {}), verify_plan.thresholds),
                                _proxy_score(
                                    x.get("metrics", {}),
                                    a_weight=proxyA_weight,
                                    a_cap=proxyA_cap,
                                    a_transform=proxyA_transform,
                                ),
                                x.get("score_hi", x["score"]),
                            ),
                        )

                if refine_enabled and is_layered and top_final:
                    if refine_targets == "holdout":
                        refine_bc = bc_hold
                        refine_interior = interior_hold
                        refine_bc_target = V_bc_hold_mid_corr if V_bc_hold_mid_corr is not None else V_bc_hold_corr
                        refine_X = X_hold
                        refine_V = V_hold_mid_corr
                        refine_is_boundary = is_boundary_hold
                    else:
                        refine_bc = bc_train
                        refine_interior = interior_train
                        refine_bc_target = V_bc_train_corr
                        refine_X = X_train
                        refine_V = V_train
                        refine_is_boundary = is_boundary

                    assert_cuda_tensor(refine_bc, "refine_bc_points")
                    assert_cuda_tensor(refine_interior, "refine_interior_points")
                    assert_cuda_tensor(refine_bc_target, "refine_bc_target")
                    assert_cuda_tensor(refine_X, "refine_X")
                    assert_cuda_tensor(refine_V, "refine_V")
                    assert_cuda_tensor(refine_is_boundary, "refine_is_boundary")

                    denom_bc = max(float(oracle_bc_mean_abs), 1e-12)
                    denom_lap = max(float(lap_denom), 1e-12)

                    source_xy = None
                    charge_locations = spec.get_charge_locations()
                    if charge_locations:
                        source_xy = torch.tensor(charge_locations[0][:2], device=device, dtype=torch.float32)
                    min_z_imag = 1e-6
                    layer_h = None
                    for layer in getattr(spec, "dielectrics", None) or []:
                        z_min = layer.get("z_min")
                        z_max = layer.get("z_max")
                        if z_min is None or z_max is None:
                            continue
                        try:
                            z_min_f = float(z_min)
                            z_max_f = float(z_max)
                        except Exception:
                            continue
                        if math.isfinite(z_min_f) and math.isfinite(z_max_f):
                            layer_h = abs(z_max_f - z_min_f)
                            break
                    if layer_h is not None and layer_h > 0:
                        min_z_imag = max(min_z_imag, 1e-6 * layer_h)

                    for cand in top_final:
                        if _has_dcim_block(cand["elements"]):
                            continue
                        if cand["metrics"].get("holdout_nonfinite"):
                            continue
                        if refine_max_terms > 0 and len(cand["elements"]) > refine_max_terms:
                            continue
                        if not (math.isfinite(denom_bc) and math.isfinite(denom_lap)):
                            continue
                        base_rel_bc = float(cand["metrics"].get("rel_bc_err_holdout", float("inf")))
                        base_rel_lap = float(cand["metrics"].get("rel_lap_holdout", float("inf")))
                        if math.isfinite(base_rel_bc) and math.isfinite(base_rel_lap):
                            base_obj = base_rel_bc + w_pde * base_rel_lap
                        else:
                            base_obj = float("inf")

                        refine_elements: List[ImageBasisElement] = []
                        refine_params: List[torch.Tensor] = []
                        fixed_xy: List[Tuple[torch.Tensor, torch.Tensor]] = []
                        refine_z_imag: List[torch.Tensor] = []

                        for elem in cand["elements"]:
                            if isinstance(elem, (DCIMPoleImageBasis, DCIMBranchCutImageBasis)):
                                params = dict(elem.params)
                                pos = params.get("position")
                                if pos is None:
                                    refine_elements.append(elem)
                                    continue
                                pos = pos.to(device=device, dtype=torch.float32).view(3).clone().detach()
                                if source_xy is not None:
                                    with torch.no_grad():
                                        pos[:2] = source_xy
                                pos.requires_grad_(True)
                                fixed = pos.detach().clone()
                                if source_xy is not None:
                                    fixed[:2] = source_xy

                                def _xy_hook(grad: torch.Tensor, fixed_xy: torch.Tensor = fixed) -> torch.Tensor:
                                    grad = grad.clone()
                                    grad[0:2] = 0.0
                                    return grad

                                pos.register_hook(_xy_hook)
                                fixed_xy.append((pos, fixed))
                                params["position"] = pos
                                assert_cuda_tensor(pos, "refine_pos")

                                z_imag = params.get("z_imag")
                                if z_imag is None:
                                    z_imag = torch.zeros((), device=device, dtype=torch.float32)
                                else:
                                    z_imag = torch.as_tensor(z_imag, device=device, dtype=torch.float32).view(())
                                z_imag = z_imag.clone().detach()
                                with torch.no_grad():
                                    z_imag.clamp_(min=min_z_imag)
                                z_imag.requires_grad_(True)
                                params["z_imag"] = z_imag
                                refine_z_imag.append(z_imag)
                                assert_cuda_tensor(z_imag, "refine_z_imag")

                                new_elem = elem.__class__(params)
                                info = getattr(elem, "_group_info", None)
                                if isinstance(info, dict):
                                    setattr(new_elem, "_group_info", dict(info))
                                refine_elements.append(new_elem)
                                refine_params.extend([pos, z_imag])
                                continue

                            if isinstance(elem, PointChargeBasis):
                                params = dict(elem.params)
                                pos = params.get("position")
                                if pos is None:
                                    refine_elements.append(elem)
                                    continue
                                pos = pos.to(device=device, dtype=torch.float32).view(3).clone().detach()
                                if source_xy is not None:
                                    with torch.no_grad():
                                        pos[:2] = source_xy
                                pos.requires_grad_(True)
                                fixed = pos.detach().clone()
                                if source_xy is not None:
                                    fixed[:2] = source_xy

                                def _xy_hook(grad: torch.Tensor, fixed_xy: torch.Tensor = fixed) -> torch.Tensor:
                                    grad = grad.clone()
                                    grad[0:2] = 0.0
                                    return grad

                                pos.register_hook(_xy_hook)
                                fixed_xy.append((pos, fixed))
                                params["position"] = pos
                                assert_cuda_tensor(pos, "refine_pos")

                                new_elem = PointChargeBasis(params, type_name=elem.type)
                                info = getattr(elem, "_group_info", None)
                                if isinstance(info, dict):
                                    setattr(new_elem, "_group_info", dict(info))
                                refine_elements.append(new_elem)
                                refine_params.append(pos)
                                continue

                            refine_elements.append(elem)

                        if not refine_params:
                            continue

                        refine_attempted_total += 1
                        refine_weights = cand["weights"].detach().to(device=device, dtype=torch.float32).clone()
                        refine_weights.requires_grad_(True)
                        refine_params.append(refine_weights)
                        assert_cuda_tensor(refine_weights, "refine_weights")
                        system = ImageSystem(refine_elements, refine_weights)

                        def _enforce_refine_invariants() -> None:
                            if fixed_xy:
                                with torch.no_grad():
                                    for pos, fixed in fixed_xy:
                                        pos[:2] = fixed[:2]
                            if refine_z_imag:
                                with torch.no_grad():
                                    for z_imag in refine_z_imag:
                                        z_imag.clamp_(min=min_z_imag)

                        def _refine_eval(pts: torch.Tensor) -> torch.Tensor:
                            assert_cuda_tensor(pts, "refine_eval_points")
                            out = system.potential(pts)
                            assert_cuda_tensor(out, "refine_eval_out")
                            return out

                        bad_refine = False

                        def _refine_objective() -> torch.Tensor:
                            nonlocal bad_refine
                            pred_bc = _refine_eval(refine_bc)
                            bc_err = torch.mean(torch.abs(pred_bc - refine_bc_target))
                            rel_bc = bc_err / denom_bc
                            lap = _laplacian_fd(_refine_eval, refine_interior, h=1e-2)
                            lap_mean = torch.mean(torch.abs(lap))
                            rel_lap = lap_mean / denom_lap
                            loss = rel_bc + w_pde * rel_lap
                            if not torch.isfinite(loss).item():
                                bad_refine = True
                                return torch.zeros((), device=device)
                            return loss

                        if refine_opt == "lbfgs":
                            optimizer = torch.optim.LBFGS(
                                refine_params,
                                lr=refine_lr,
                                max_iter=max(1, refine_steps),
                            )

                            def _closure() -> torch.Tensor:
                                optimizer.zero_grad(set_to_none=True)
                                loss = _refine_objective()
                                loss.backward()
                                return loss

                            optimizer.step(_closure)
                            _enforce_refine_invariants()
                            if bad_refine:
                                continue
                        else:
                            optimizer = torch.optim.Adam(refine_params, lr=refine_lr)
                            for _ in range(max(1, refine_steps)):
                                optimizer.zero_grad(set_to_none=True)
                                loss = _refine_objective()
                                if bad_refine:
                                    break
                                loss.backward()
                                optimizer.step()
                                _enforce_refine_invariants()
                            if bad_refine:
                                continue

                        _enforce_refine_invariants()

                        A_refine = assemble_basis_matrix(refine_elements, refine_X)
                        assert_cuda_tensor(A_refine, "A_refine")
                        refine_V_fit = refine_V
                        refine_is_boundary_fit = refine_is_boundary
                        if interface_constraint_data is not None:
                            A_refine, refine_V_fit, refine_is_boundary_fit = _apply_interface_constraints(
                                A_train=A_refine,
                                V_train=refine_V_fit,
                                is_boundary=refine_is_boundary_fit,
                                elements=refine_elements,
                                constraint_data=interface_constraint_data,
                            )
                        A_refine_scaled, col_norms = _scale_columns(A_refine)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        weights_scaled, _ = solve_sparse(
                            A_refine_scaled,
                            refine_X,
                            refine_V_fit,
                            refine_is_boundary_fit,
                            _NullLogger(),
                            reg_l1=float(solver_cfg.get("reg_l1", 1e-3)),
                            solver=solver_mode,
                            max_iter=int(solver_cfg.get("max_iters", 64)),
                            tol=float(solver_cfg.get("tol", 1e-6)),
                            lambda_group=float(solver_cfg.get("lambda_group", 0.0)),
                            normalize_columns=False,
                        )
                        refined_weights = weights_scaled / col_norms
                        if refined_weights.numel() == 0:
                            continue
                        assert_cuda_tensor(refined_weights, "refined_weights")
                        weights_ok, reason = _validate_weights(refined_weights)
                        if not weights_ok:
                            _reject_invalid_weights(
                                gen_idx=gen,
                                program_idx=-1,
                                program=cand["program"],
                                elements=refine_elements,
                                A_train=A_refine,
                                V_train=refine_V_fit,
                                weights=refined_weights,
                                pred_hold=None,
                                reason=reason,
                            )
                            continue
                        end.record()
                        torch.cuda.synchronize()
                        t_solve_ms = float(start.elapsed_time(end))

                        system_refined = ImageSystem(refine_elements, refined_weights)

                        def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                            assert_cuda_tensor(pts, "refined_eval_points")
                            with torch.no_grad():
                                out = system_refined.potential(pts)
                                if use_ref:
                                    out = out + compute_layered_reference_potential(
                                        spec,
                                        pts,
                                        device=pts.device,
                                        dtype=pts.dtype,
                                    )
                            assert_cuda_tensor(out, "refined_eval_out")
                            return out

                        with torch.no_grad():
                            A_hold = assemble_basis_matrix(refine_elements, X_hold)
                            assert_cuda_tensor(A_hold, "A_hold_refined")
                            pred_hold_corr = A_hold.matmul(refined_weights)
                            pred_hold = _add_reference(pred_hold_corr, V_ref_hold)
                            bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                            in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                            mean_bc = _mean_safe(bc_err)
                            mean_in = _mean_safe(in_err)
                            max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                            max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                            mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            mean_bc_mid = _mean_safe(mid_bc_err)
                            mean_in_mid = _mean_safe(mid_in_err)
                            rel_bc = mean_bc_mid / denom_bc
                            rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                            lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                            lap_abs = torch.abs(lap)
                            lap_mean = _mean_safe(lap_abs)
                            lap_max = float(torch.max(lap_abs).item()) if lap_abs.numel() else 0.0
                            rel_lap = lap_mean / denom_lap

                        eval_ms = _timed_cuda(_eval_fn, interior_hold, warmup=1, repeat=3)

                        if not (math.isfinite(rel_bc) and math.isfinite(rel_lap)):
                            continue

                        perturbed = _perturb_elements(refine_elements, stability_sigma, device=device)
                        stability_ratio = float("nan")
                        if perturbed is not None:
                            A_hold_pert = assemble_basis_matrix(perturbed, X_hold)
                            assert_cuda_tensor(A_hold_pert, "A_hold_refined_pert")
                            pert_pred_corr = A_hold_pert.matmul(refined_weights)
                            pert_pred = _add_reference(pert_pred_corr, V_ref_hold)
                            base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                            pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            pert_err = w_bc * _mean_safe(pert_bc) + w_pde * _mean_safe(pert_in)
                            stability_ratio = float(pert_err / max(base_err, 1e-8))

                        comp = _complexity(cand["program"], refine_elements)
                        complex_count = _count_complex_terms(refine_elements, device=device)
                        metrics = {
                            "max_bc_err_holdout": max_bc,
                            "mean_bc_err_holdout": mean_bc,
                            "max_pde_err_holdout": lap_max,
                            "mean_pde_err_holdout": lap_mean,
                            "mid_bc_mean_abs": mean_bc_mid,
                            "mid_pde_mean_abs": mean_in_mid,
                            "rel_bc_err_holdout": float(rel_bc),
                            "rel_pde_err_holdout": float(rel_in),
                            "lap_mean_abs_holdout": float(lap_mean),
                            "rel_lap_holdout": float(rel_lap),
                            "oracle_bc_mean_abs_holdout": float(oracle_bc_mean_abs),
                            "oracle_in_mean_abs_holdout": float(oracle_in_mean_abs),
                            "stability_ratio": stability_ratio,
                            "eval_time_us": float(eval_ms * 1000.0) if eval_ms == eval_ms else float("nan"),
                            "solve_time_us": float(t_solve_ms * 1000.0) if t_solve_ms == t_solve_ms else float("nan"),
                            "total_time_us": float((eval_ms + t_solve_ms) * 1000.0) if eval_ms == eval_ms else float("nan"),
                            "complexity_terms": comp["n_terms"],
                            "complexity_nodes": comp["n_nodes"],
                            "n_terms": comp["n_terms"],
                            "complex_count": int(complex_count),
                            "refine_applied": True,
                        }
                        score_mid = (
                            w_bc * rel_bc
                            + w_pde * rel_lap
                            + w_complexity * comp["n_terms"]
                            + w_latency * (metrics["eval_time_us"] / 1e6)
                            + w_stability * stability_ratio
                        )

                        rel_bc_val = float(rel_bc)
                        rel_lap_val = float(rel_lap)
                        new_obj = rel_bc_val + w_pde * rel_lap_val
                        rel_bc_ok = math.isfinite(rel_bc_val) and (
                            not math.isfinite(base_rel_bc) or rel_bc_val <= base_rel_bc
                        )
                        rel_lap_ok = math.isfinite(rel_lap_val) and (
                            not math.isfinite(base_rel_lap) or rel_lap_val <= base_rel_lap
                        )
                        if new_obj < base_obj and rel_bc_ok and rel_lap_ok:
                            def _detach_params(params: Dict[str, Any]) -> Dict[str, Any]:
                                return {
                                    k: (v.detach() if torch.is_tensor(v) else v)
                                    for k, v in params.items()
                                }

                            refined_elements_out: List[ImageBasisElement] = []
                            for elem in refine_elements:
                                new_elem = elem
                                if isinstance(elem, (DCIMPoleImageBasis, DCIMBranchCutImageBasis)):
                                    new_elem = elem.__class__(_detach_params(elem.params))
                                elif isinstance(elem, PointChargeBasis):
                                    new_elem = PointChargeBasis(_detach_params(elem.params), type_name=elem.type)
                                if new_elem is not elem:
                                    info = getattr(elem, "_group_info", None)
                                    if isinstance(info, dict):
                                        setattr(new_elem, "_group_info", dict(info))
                                refined_elements_out.append(new_elem)

                            cand["elements"] = refined_elements_out
                            cand["weights"] = refined_weights.detach()
                            cand["metrics"] = metrics
                            cand["score"] = float(score_mid)
                            refine_improved_total += 1
                            if best_refined_rel_bc is None or rel_bc_val < best_refined_rel_bc:
                                best_refined_rel_bc = rel_bc_val
                            if best_refined_rel_lap is None or rel_lap_val < best_refined_rel_lap:
                                best_refined_rel_lap = rel_lap_val

                for rank, cand in enumerate(top_final):
                    weights_ok, reason = _validate_weights(cand["weights"])
                    if not weights_ok:
                        _reject_invalid_weights(
                            gen_idx=gen,
                            program_idx=rank,
                            program=cand["program"],
                            elements=cand["elements"],
                            A_train=None,
                            V_train=None,
                            weights=cand["weights"],
                            pred_hold=None,
                            reason=reason,
                        )
                        continue
                    system = ImageSystem(cand["elements"], cand["weights"])

                    def _verify_eval(pts: torch.Tensor) -> torch.Tensor:
                        assert_cuda_tensor(pts, "verifier_points")
                        out = system.potential(pts)
                        if use_ref:
                            out = out + compute_layered_reference_potential(
                                spec,
                                pts,
                                device=pts.device,
                                dtype=pts.dtype,
                            )
                        assert_cuda_tensor(out, "verifier_out")
                        return out

                    rec = _candidate_to_record(
                        gen,
                        rank,
                        cand["program"],
                        cand["elements"],
                        cand["weights"],
                        cand["metrics"],
                        cand["score"],
                    )
                    holdout_ok = not cand["metrics"].get("holdout_nonfinite")
                    mean_bc = float(cand["metrics"].get("mean_bc_err_holdout", float("inf")))
                    if holdout_ok and (best_mean_bc is None or mean_bc < best_mean_bc):
                        best_mean_bc = mean_bc
                    rel_bc_val = float(cand["metrics"].get("rel_bc_err_holdout", float("inf")))
                    if holdout_ok and math.isfinite(rel_bc_val) and (best_rel_bc is None or rel_bc_val < best_rel_bc):
                        best_rel_bc = rel_bc_val
                        best_rel_in = float(cand["metrics"].get("rel_pde_err_holdout", float("inf")))
                        best_rel_elements = cand["elements"]
                        best_rel_metrics = cand["metrics"]
                    rel_in_val = float(cand["metrics"].get("rel_pde_err_holdout", float("inf")))
                    if holdout_ok and math.isfinite(rel_in_val) and (best_rel_in is None or rel_in_val < best_rel_in):
                        best_rel_in = rel_in_val
                        best_rel_in_metrics = cand["metrics"]
                    rel_lap_val = float(cand["metrics"].get("rel_lap_holdout", float("inf")))
                    if holdout_ok and math.isfinite(rel_lap_val) and (best_rel_lap is None or rel_lap_val < best_rel_lap):
                        best_rel_lap = rel_lap_val
                        best_rel_lap_metrics = cand["metrics"]

                    cert_dir = run_dir / "artifacts" / "certificates" / f"gen{gen:03d}_rank{rank}_verifier"
                    cert_status = "error"
                    cert_error = None
                    _count_preflight("verified_attempted")
                    try:
                        certificate = verifier.run(
                            {"eval_fn": _verify_eval},
                            spec.to_json(),
                            verify_plan,
                            points=interior_hold,
                            outdir=cert_dir,
                        )
                        cert_status = certificate.final_status
                    except Exception as exc:
                        cert_error = str(exc)
                    rec["verification"] = {"status": cert_status, "path": str(cert_dir)}
                    if cert_error:
                        rec["verification"]["error"] = cert_error

                    append_jsonl(best_path, rec)
                    best_records.append(rec)

                    cert_payload = {
                        "timestamp": utc_now_iso(),
                        "spec_digest": sha256_json(spec.to_json()),
                        "candidate_digest": sha256_json(rec),
                        "generation": int(gen),
                        "rank": int(rank),
                        "program": rec["program"],
                        "elements": rec["elements"],
                        "weights": rec["weights"],
                        "metrics": rec["metrics"],
                        "verification": rec.get("verification", {}),
                    }
                    cert_path = run_dir / "artifacts" / "certificates" / f"gen{gen:03d}_rank{rank}_summary.json"
                    write_json(cert_path, cert_payload)
                    _count_preflight("verified_written")

                gen_rel_bc = float("inf")
                gen_rel_in = float("inf")
                gen_rel_lap = float("inf")
                for cand in top_final:
                    if cand["metrics"].get("holdout_nonfinite"):
                        continue
                    rel_bc_val = cand["metrics"].get("rel_bc_err_holdout")
                    rel_in_val = cand["metrics"].get("rel_pde_err_holdout")
                    rel_lap_val = cand["metrics"].get("rel_lap_holdout")
                    if rel_bc_val is not None and math.isfinite(float(rel_bc_val)):
                        gen_rel_bc = min(gen_rel_bc, float(rel_bc_val))
                    if rel_in_val is not None and math.isfinite(float(rel_in_val)):
                        gen_rel_in = min(gen_rel_in, float(rel_in_val))
                    if rel_lap_val is not None and math.isfinite(float(rel_lap_val)):
                        gen_rel_lap = min(gen_rel_lap, float(rel_lap_val))
                if gen_rel_bc == float("inf"):
                    gen_rel_bc = float("nan")
                if gen_rel_in == float("inf"):
                    gen_rel_in = float("nan")
                if gen_rel_lap == float("inf"):
                    gen_rel_lap = float("nan")
                per_gen_rel_bc.append(gen_rel_bc)
                per_gen_rel_in.append(gen_rel_in)
                per_gen_rel_lap.append(gen_rel_lap)
                if preflight_enabled and gen_counters is not None:
                    entry = {"gen": int(gen), "counters": gen_counters.as_dict()}
                    if holdout_stats:
                        entry["holdout"] = dict(holdout_stats)
                    entry["fraction_complex_candidates"] = float(gen_fraction_complex)
                    entry["fraction_dcim_candidates"] = float(gen_fraction_dcim)
                    if gen < len(per_gen_preflight):
                        per_gen_preflight[gen] = entry
                    else:
                        per_gen_preflight.append(entry)
                    run_logger.info(
                        "PREFLIGHT gen=%s compiled_ok=%s solved_ok=%s fast_scored=%s verified_written=%s",
                        gen,
                        gen_counters.compiled_ok,
                        gen_counters.solved_ok,
                        gen_counters.fast_scored,
                        gen_counters.verified_written,
                    )
                    _write_preflight_snapshot()
                if is_layered and preflight_enabled and gen_candidates_total > 0:
                    if gen_fraction_complex < 0.30:
                        low_complex_streak += 1
                    else:
                        low_complex_streak = 0
                    if gen_fraction_dcim < 0.30:
                        low_dcim_streak += 1
                    else:
                        low_dcim_streak = 0
                    if low_complex_streak >= 3 or low_dcim_streak >= 3:
                        raise RuntimeError(
                            "DCIM/complex fraction below 0.30 for 3 consecutive generations; "
                            "inspect candidate emission and preflight counters."
                        )

                if ramp_check:
                    improved = False
                    if gen_rel_bc == gen_rel_bc:
                        if ramp_best_rel_bc is None:
                            ramp_best_rel_bc = gen_rel_bc
                            improved = True
                        else:
                            target = ramp_best_rel_bc * (1.0 - ramp_min_rel_improve)
                            if gen_rel_bc < target:
                                ramp_best_rel_bc = gen_rel_bc
                                improved = True
                    if gen_rel_lap == gen_rel_lap:
                        if ramp_best_rel_lap is None:
                            ramp_best_rel_lap = gen_rel_lap
                            improved = True
                        else:
                            target = ramp_best_rel_lap * (1.0 - ramp_min_rel_improve)
                            if gen_rel_lap < target:
                                ramp_best_rel_lap = gen_rel_lap
                                improved = True
                    if improved:
                        last_improve_gen = gen
                    if (
                        not ramp_abort
                        and (ramp_best_rel_bc is not None or ramp_best_rel_lap is not None)
                        and gen - last_improve_gen >= ramp_patience
                    ):
                        run_logger.warning("RAMP ABORT: not improving")
                        ramp_abort = True
                        if should_early_exit(allow_not_ready, ramp_abort):
                            break

                empty_count = sum(1 for m in fast_metrics if m.get("empty"))
                empty_frac = empty_count / max(1, len(fast_metrics))
                per_gen_empty_frac.append(float(empty_frac))

                fast_stats = {
                    "min": float(min(fast_scores)),
                    "mean": float(sum(fast_scores) / max(len(fast_scores), 1)),
                    "max": float(max(fast_scores)),
                }
                append_jsonl(
                    metrics_path,
                    {
                        "generation": gen,
                        "spec_hash": spec_hash,
                        "population_B": backoff.population_B,
                        "points": {
                            "bc_train": backoff.bc_train,
                            "interior_train": backoff.interior_train,
                            "bc_holdout": backoff.bc_holdout,
                            "interior_holdout": backoff.interior_holdout,
                        },
                        "fast_scores": fast_stats,
                        "topk_fast": topK_fast,
                        "topk_mid": topk_mid,
                        "topk_final": topk_final,
                        "fast_empty_frac": float(empty_frac),
                        "gen_time_s": float(time.perf_counter() - start_gen),
                    },
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    retry = True
                    if backoff.population_B > 64:
                        backoff.population_B = max(64, backoff.population_B // 2)
                    elif backoff.bc_train > 256:
                        backoff.bc_train = max(256, backoff.bc_train // 2)
                        backoff.bc_holdout = max(256, backoff.bc_holdout // 2)
                        backoff.interior_train = max(256, backoff.interior_train // 2)
                        backoff.interior_holdout = max(256, backoff.interior_holdout // 2)
                    else:
                        raise
                else:
                    raise
            if should_early_exit(allow_not_ready, ramp_abort):
                break
        if should_early_exit(allow_not_ready, ramp_abort):
            break

    best_bc_rel = float("nan")
    best_lap_rel = float("nan")
    if best_rel_metrics is not None:
        bc_abs = float(best_rel_metrics.get("mid_bc_mean_abs", best_rel_metrics.get("mean_bc_err_holdout", float("nan"))))
        bc_rel = float(best_rel_metrics.get("rel_bc_err_holdout", float("nan")))
        best_bc_rel = bc_rel
        in_metrics = best_rel_in_metrics or best_rel_metrics
        in_abs = float(in_metrics.get("mid_pde_mean_abs", in_metrics.get("mean_pde_err_holdout", float("nan"))))
        in_rel = float(
            best_rel_in
            if best_rel_in is not None
            else in_metrics.get("rel_pde_err_holdout", float("nan"))
        )
        lap_metrics = best_rel_lap_metrics or best_rel_metrics
        lap_abs = float(lap_metrics.get("lap_mean_abs_holdout", lap_metrics.get("mean_pde_err_holdout", float("nan"))))
        lap_rel = float(lap_metrics.get("rel_lap_holdout", float("nan")))
        best_lap_rel = lap_rel
        run_logger.info(
            "RUN SUMMARY: best_bc_abs=%.3e best_bc_rel=%.3e best_in_abs=%.3e best_in_rel=%.3e "
            "best_lap_abs=%.3e best_lap_rel=%.3e",
            bc_abs,
            bc_rel,
            in_abs,
            in_rel,
            lap_abs,
            lap_rel,
        )
    else:
        run_logger.info("RUN SUMMARY: no candidates evaluated")

    run_logger.info("RUN DIR: %s", run_dir)
    run_logger.info("ENV JSON: %s", run_dir / "env.json")

    def _metric_float(metrics: Dict[str, Any], key: str, fallback: str | None = None) -> float:
        if key in metrics:
            val = metrics.get(key)
        elif fallback is not None:
            val = metrics.get(fallback)
        else:
            val = None
        try:
            return float(val)
        except Exception:
            return float("nan")

    if best_records:
        def _sort_rel(rec: Dict[str, Any]) -> float:
            rel = _metric_float(rec.get("metrics", {}), "rel_bc_err_holdout")
            return rel if rel == rel else float("inf")

        top_records = sorted(best_records, key=_sort_rel)[:5]
        run_logger.info("TOP 5 CANDIDATES (by rel_bc):")
        for idx, rec in enumerate(top_records, start=1):
            metrics = rec.get("metrics", {})
            mean_bc = _metric_float(metrics, "mid_bc_mean_abs", "mean_bc_err_holdout")
            rel_bc = _metric_float(metrics, "rel_bc_err_holdout")
            mean_in = _metric_float(metrics, "mid_pde_mean_abs", "mean_pde_err_holdout")
            rel_in = _metric_float(metrics, "rel_pde_err_holdout")
            rel_lap = _metric_float(metrics, "rel_lap_holdout")
            complex_count = int(metrics.get("complex_count", 0) or 0)
            n_terms = int(metrics.get("n_terms", metrics.get("complexity_terms", 0)) or 0)
            dcim_marker = " [DCIM_BASELINE]" if metrics.get("is_dcim_block_baseline") else ""
            run_logger.info(
                "%02d mean_bc=%.3e rel_bc=%.3e mean_in=%.3e rel_in=%.3e rel_lap=%.3e "
                "complex_count=%s n_terms=%s%s",
                idx,
                mean_bc,
                rel_bc,
                mean_in,
                rel_in,
                rel_lap,
                complex_count,
                n_terms,
                dcim_marker,
            )
    else:
        run_logger.info("TOP 5 CANDIDATES: none")

    if best_rel_elements:
        hist = _element_type_hist(best_rel_elements)
        hist_str = ", ".join(f"{k}: {v}" for k, v in hist.items())
        dcim_present = bool(best_rel_metrics and best_rel_metrics.get("is_dcim_block_baseline"))
        dcim_present = dcim_present or _has_dcim_block(best_rel_elements)
        run_logger.info("BEST ELEMENT TYPES: %s | dcim_block_present=%s", hist_str, dcim_present)
    else:
        run_logger.info("BEST ELEMENT TYPES: none")

    dcim_used_as_best = _dcim_used_as_best(best_rel_metrics, best_rel_elements)
    if best_dcim_rel_bc is None and best_dcim_rel_lap is None and best_dcim_score is None:
        run_logger.info("DCIM BASELINE BEST: unavailable")
    else:
        dcim_rel_bc = best_dcim_rel_bc if best_dcim_rel_bc is not None else float("nan")
        dcim_rel_lap = best_dcim_rel_lap if best_dcim_rel_lap is not None else float("nan")
        dcim_score = best_dcim_score if best_dcim_score is not None else float("nan")
        run_logger.info(
            "DCIM BASELINE BEST: rel_bc=%.3e rel_lap=%.3e score=%.3e used_as_best=%s",
            dcim_rel_bc,
            dcim_rel_lap,
            dcim_score,
            dcim_used_as_best,
        )

    def _first_finite(vals: Sequence[float]) -> float:
        for val in vals:
            if val == val and math.isfinite(val):
                return float(val)
        return float("nan")

    def _last_finite(vals: Sequence[float]) -> float:
        for val in reversed(vals):
            if val == val and math.isfinite(val):
                return float(val)
        return float("nan")

    first_rel_bc = _first_finite(per_gen_rel_bc)
    last_rel_bc = _last_finite(per_gen_rel_bc)
    first_rel_lap = _first_finite(per_gen_rel_lap)
    last_rel_lap = _last_finite(per_gen_rel_lap)
    improve_thresh = max(0.0, ramp_min_rel_improve)
    improved_bc = (
        first_rel_bc == first_rel_bc
        and last_rel_bc == last_rel_bc
        and last_rel_bc < first_rel_bc * (1.0 - improve_thresh)
    )
    improved_lap = (
        first_rel_lap == first_rel_lap
        and last_rel_lap == last_rel_lap
        and last_rel_lap < first_rel_lap * (1.0 - improve_thresh)
    )
    final_rel_bc = last_rel_bc
    ready = improved_bc and improved_lap and final_rel_bc == final_rel_bc and final_rel_bc < 1e-3
    spec_hash_report = "none"
    spec_hash_constant = None
    if spec_hashes:
        unique_hashes = sorted(set(spec_hashes))
        if fixed_spec:
            spec_hash_report = unique_hashes[0]
            spec_hash_constant = len(unique_hashes) == 1
        else:
            spec_hash_report = unique_hashes[-1]
    if fixed_spec:
        status = "constant" if spec_hash_constant else "varied"
        run_logger.info("SPEC HASH: %s (%s)", spec_hash_report, status)
    else:
        run_logger.info("SPEC HASH: %s", spec_hash_report)
    run_logger.info(
        "RAMP SIGNAL: improved_bc=%s improved_lap=%s best_bc_rel=%.3e best_lap_rel=%.3e ready=%s",
        improved_bc,
        improved_lap,
        best_bc_rel,
        best_lap_rel,
        ready,
    )
    if ready:
        run_logger.info("READY for monster run")
    else:
        limiter = None
        if per_gen_empty_frac and per_gen_empty_frac[-1] >= 0.5:
            limiter = "too many empties"
        elif not (improved_bc and improved_lap):
            limiter = "no rel improvement"
        elif final_rel_bc != final_rel_bc or final_rel_bc >= 1e-3:
            limiter = "rel_bc above threshold"
        else:
            limiter = "timing too slow"
        run_logger.info("NOT READY: %s", limiter)
    if refine_enabled:
        run_logger.info("REFINE IMPROVED: %s/%s", refine_improved_total, refine_attempted_total)
        if refine_improved_total > 0:
            ref_bc = (
                best_refined_rel_bc
                if best_refined_rel_bc is not None
                else float("nan")
            )
            ref_lap = (
                best_refined_rel_lap
                if best_refined_rel_lap is not None
                else float("nan")
            )
            run_logger.info("REFINE BEST REL: rel_bc=%.3e rel_lap=%.3e", ref_bc, ref_lap)

    if preflight_counters is not None:
        _write_preflight_snapshot()
        summarize_to_stdout(preflight_counters)
        run_logger.info("%s", _format_preflight_summary(preflight_counters))

    if should_early_exit(allow_not_ready, ramp_abort):
        return 3

    if sanity_threshold is not None:
        if best_mean_bc is None or best_mean_bc > sanity_threshold:
            run_logger.info(
                "Sanity FAIL: best mean BC holdout %s exceeds threshold %s.",
                best_mean_bc,
                sanity_threshold,
            )
            return 2
        run_logger.info(
            "Sanity PASS: best mean BC holdout %s within threshold %s.",
            best_mean_bc,
            sanity_threshold,
        )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Electrodrive discovery experiment runner.")
    parser.add_argument("--config", required=True, help="Path to discovery YAML config.")
    parser.add_argument("--debug", action="store_true", help="Enable debug reductions and CUDA_LAUNCH_BLOCKING.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_discovery(Path(args.config), debug=bool(args.debug))


if __name__ == "__main__":
    raise SystemExit(main())
