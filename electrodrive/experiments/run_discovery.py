from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
)
from electrodrive.images.geo_encoder import GeoEncoder
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.images.search import ImageSystem, assemble_basis_matrix, solve_sparse
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.verify.oracle_backends.f0 import F0AnalyticOracleBackend
from electrodrive.verify.oracle_backends.f1_sommerfeld import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuery, OracleQuantity
from electrodrive.verify.utils import normalize_dtype
from electrodrive.verify.utils import sha256_json, utc_now_iso
from electrodrive.verify.verifier import VerificationPlan, Verifier


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


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


def _set_perf_flags(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    use_tf32 = bool(model_cfg.get("use_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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


def _fast_weights(A: torch.Tensor, b: torch.Tensor, reg: float) -> torch.Tensor:
    if A.numel() == 0:
        return torch.zeros((0,), device=A.device, dtype=A.dtype)
    k = A.shape[1]
    if k == 0:
        return torch.zeros((0,), device=A.device, dtype=A.dtype)
    A_scaled, col_norms = _scale_columns(A)
    ata = A_scaled.transpose(0, 1).matmul(A_scaled)
    ata = ata + reg * torch.eye(k, device=A.device, dtype=A.dtype)
    atb = A_scaled.transpose(0, 1).matmul(b)
    try:
        w_scaled = torch.linalg.solve(ata, atb)
    except RuntimeError:
        try:
            w_scaled = torch.linalg.lstsq(A_scaled, b).solution
        except RuntimeError:
            return torch.zeros((k,), device=A.device, dtype=A.dtype)
    return w_scaled / col_norms


def _scale_columns(A: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.numel() == 0:
        return A, torch.ones((A.shape[1],), device=A.device, dtype=A.dtype)
    col_norms = torch.linalg.norm(A, dim=0).clamp_min(eps)
    return A / col_norms, col_norms


def _mean_abs(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(x)).item())


def _laplacian_denom(oracle_in_mean_abs: float, oracle_bc_mean_abs: float) -> float:
    return max(float(oracle_in_mean_abs), 1e-6 * float(oracle_bc_mean_abs), 1e-12)


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
) -> List[Program]:
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
    run_dir = _make_run_dir(tag)
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
        flow_cfg = FlowConfig(
            n_steps=int(param_cfg.get("steps", 4)),
            solver=str(param_cfg.get("solver", "euler")),
            temperature=float(param_cfg.get("temperature", 1.0)),
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
    ramp_patience = int(run_cfg.get("ramp_patience_gens", 3))
    ramp_min_rel_improve = float(run_cfg.get("ramp_min_rel_improvement", 0.10))
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

    fixed_spec_obj: Optional[CanonicalSpec] = None
    fixed_spec_meta: Optional[Dict[str, Any]] = None
    if fixed_spec:
        for _ in range(fixed_spec_index + 1):
            fixed_spec_obj = sampler.sample()
        if fixed_spec_obj is None:
            raise RuntimeError("fixed_spec enabled but no spec could be sampled")
        fixed_spec_meta = _spec_metadata_from_spec(fixed_spec_obj)

    for gen in range(generations):
        start_gen = time.perf_counter()
        if fixed_spec and fixed_spec_obj is not None:
            spec = fixed_spec_obj
            spec_meta = fixed_spec_meta or _spec_metadata_from_spec(spec)
        else:
            spec = sampler.sample()
            spec_meta = _spec_metadata_from_spec(spec)
        spec_hash = sha256_json(spec.to_json())
        spec_hashes.append(spec_hash)
        domain_scale = float(spec_cfg.get("domain_scale", 1.0))
        seed_gen = seed + gen * 13
        is_layered = getattr(spec, "BCs", "") == "dielectric_interfaces"

        torch.cuda.synchronize()
        retry = True
        attempt = 0
        while retry:
            attempt += 1
            retry = False
            try:
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
                oracle_bc_mean_abs = _mean_abs(V_bc_hold_mid) if V_bc_hold_mid is not None else _mean_abs(V_bc_hold)
                oracle_in_mean_abs = _mean_abs(V_in_hold_mid) if V_in_hold_mid is not None else _mean_abs(V_in_hold)
                lap_denom = _laplacian_denom(oracle_in_mean_abs, oracle_bc_mean_abs)

                X_train = torch.cat([bc_train, interior_train], dim=0)
                V_train = torch.cat([V_bc_train, V_in_train], dim=0)
                is_boundary = torch.cat(
                    [torch.ones(bc_train.shape[0], device=device, dtype=torch.bool),
                     torch.zeros(interior_train.shape[0], device=device, dtype=torch.bool)],
                    dim=0,
                )
                X_hold = torch.cat([bc_hold, interior_hold], dim=0)
                V_hold = torch.cat([V_bc_hold, V_in_hold], dim=0)
                V_hold_mid = torch.cat([V_bc_hold_mid, V_in_hold_mid], dim=0)
                is_boundary_hold = torch.cat(
                    [torch.ones(bc_hold.shape[0], device=device, dtype=torch.bool),
                     torch.zeros(interior_hold.shape[0], device=device, dtype=torch.bool)],
                    dim=0,
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
                seeded = _seed_layered_templates(spec, max_steps=max_steps, count=8)
                if seeded:
                    programs.extend(seeded)

                if not programs:
                    raise RuntimeError("No programs sampled; aborting generation.")
                program_lengths = [len(getattr(p, "nodes", []) or []) for p in programs]
                payload = None
                if use_param_sampler:
                    program_batch = gfn.param_sampler.build_program_batch(
                        programs,
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
                        max_tokens=gfn.flow_config.max_tokens,
                        max_ast_len=gfn.flow_config.max_ast_len,
                    )

                fast_scores: List[float] = []
                fast_metrics: List[Dict[str, Any]] = []
                n_terms_list: List[int] = []
                complex_counts: List[int] = []
                hold_cache: Dict[int, torch.Tensor] = {} if cache_hold_matrices else {}
                for start in range(0, len(programs), score_microbatch):
                    end = min(len(programs), start + score_microbatch)
                    for idx in range(start, end):
                        program = programs[idx]
                        if payload is not None:
                            per_payload = payload.for_program(idx)
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
                        if not elements:
                            fast_scores.append(float("inf"))
                            fast_metrics.append({"empty": True, "complex_count": 0, "frac_complex": 0.0})
                            n_terms_list.append(0)
                            complex_counts.append(0)
                            continue
                        complex_count = _count_complex_terms(elements, device=device)
                        n_terms = int(len(elements))
                        frac_complex = float(complex_count) / max(1, n_terms)
                        complex_counts.append(complex_count)
                        if is_layered and not (complex_count >= 2 or frac_complex >= 0.25):
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
                        A_train = assemble_basis_matrix(elements, X_train)
                        A_hold = assemble_basis_matrix(elements, X_hold)
                        assert_cuda_tensor(A_train, "A_train_fast")
                        assert_cuda_tensor(A_hold, "A_hold_fast")
                        if cache_hold_matrices:
                            hold_cache[idx] = A_hold
                        weights = _fast_weights(A_train, V_train, reg=float(solver_cfg.get("reg_l1", 1e-3)))
                        pred_hold = A_hold.matmul(weights)
                        bc_err = torch.mean(torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])).item()
                        in_err = torch.mean(torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])).item()
                        comp = _complexity(program, elements)
                        score = w_bc * bc_err + w_pde * in_err + w_complexity * comp["n_terms"]
                        fast_scores.append(float(score))
                        n_terms_list.append(int(comp["n_terms"]))
                        fast_metrics.append(
                            {
                                "bc_mean_abs": float(bc_err),
                                "pde_mean_abs": float(in_err),
                                "n_terms": comp["n_terms"],
                                "program_hash": meta.get("program_hash"),
                                "complex_count": int(complex_count),
                                "frac_complex": float(frac_complex),
                            }
                        )

                if gen == 0 and programs:
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
                        print(msg)
                        raise RuntimeError(msg)
                    if is_layered:
                        zero_complex = sum(1 for c in complex_counts if c == 0)
                        frac_zero = zero_complex / max(1, len(complex_counts))
                        if frac_zero > 0.80:
                            msg = (
                                "Complex-image guard failed: z_imag stayed zero; "
                                f"compiler mapping or point eval not active (zero_complex={frac_zero:.2%})."
                            )
                            print(msg)
                            raise RuntimeError(msg)

                top_fast_idx = _select_topk(fast_scores, topK_fast)
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
                        A_train_scaled, col_norms = _scale_columns(A_train_dcim)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        weights_scaled, _ = solve_sparse(
                            A_train_scaled,
                            X_train,
                            V_train,
                            is_boundary,
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
                            end.record()
                            torch.cuda.synchronize()
                            t_solve_ms = float(start.elapsed_time(end))

                            system = ImageSystem(dcim_elements, weights)

                            A_hold_dcim = assemble_basis_matrix(dcim_elements, X_hold)
                            assert_cuda_tensor(A_hold_dcim, "A_hold_dcim")
                            pred_hold = A_hold_dcim.matmul(weights)
                            bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                            in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                            mean_bc = float(torch.mean(bc_err).item()) if bc_err.numel() else 0.0
                            mean_in = float(torch.mean(in_err).item()) if in_err.numel() else 0.0
                            max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                            max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                            mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            mean_bc_mid = float(torch.mean(mid_bc_err).item()) if mid_bc_err.numel() else 0.0
                            mean_in_mid = float(torch.mean(mid_in_err).item()) if mid_in_err.numel() else 0.0
                            rel_bc = mean_bc_mid / max(oracle_bc_mean_abs, 1e-12)
                            rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                            def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                                assert_cuda_tensor(pts, "candidate_eval_points")
                                out = system.potential(pts)
                                assert_cuda_tensor(out, "candidate_eval_out")
                                return out

                            lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                            lap_abs = torch.abs(lap)
                            lap_mean = float(torch.mean(lap_abs).item()) if lap_abs.numel() else 0.0
                            lap_max = float(torch.max(lap_abs).item()) if lap_abs.numel() else 0.0
                            rel_lap = lap_mean / lap_denom

                            eval_ms = _timed_cuda(_eval_fn, interior_hold, warmup=1, repeat=3)

                            perturbed = _perturb_elements(dcim_elements, stability_sigma, device=device)
                            stability_ratio = float("nan")
                            if perturbed is not None:
                                A_hold_pert = assemble_basis_matrix(perturbed, X_hold)
                                assert_cuda_tensor(A_hold_pert, "A_hold_dcim_pert")
                                pert_pred = A_hold_pert.matmul(weights)
                                base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                                pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                                pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                                pert_err = w_bc * float(torch.mean(pert_bc).item()) + w_pde * float(torch.mean(pert_in).item())
                                stability_ratio = float(pert_err / max(base_err, 1e-8))
                            if not math.isfinite(stability_ratio):
                                stability_ratio = 1.0

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
                                "eval_time_us": float(eval_ms * 1000.0) if eval_ms == eval_ms else float("nan"),
                                "solve_time_us": float(t_solve_ms * 1000.0) if t_solve_ms == t_solve_ms else float("nan"),
                                "total_time_us": float((eval_ms + t_solve_ms) * 1000.0) if eval_ms == eval_ms else float("nan"),
                                "complexity_terms": comp["n_terms"],
                                "complexity_nodes": comp["n_nodes"],
                                "n_terms": comp["n_terms"],
                                "complex_count": int(complex_count),
                                "is_dcim_block_baseline": True,
                                "dcim_block_weight": float(dcim_block_weight),
                                "candidate_name": "dcim_block_baseline",
                            }
                            score_mid = (
                                w_bc * rel_bc
                                + w_pde * rel_lap
                                + w_complexity * comp["n_terms"]
                                + w_latency * (metrics["eval_time_us"] / 1e6)
                                + w_stability * stability_ratio
                            )
                            if math.isfinite(float(rel_bc)) and (
                                best_dcim_rel_bc is None or float(rel_bc) < best_dcim_rel_bc
                            ):
                                best_dcim_rel_bc = float(rel_bc)
                            if math.isfinite(float(rel_lap)) and (
                                best_dcim_rel_lap is None or float(rel_lap) < best_dcim_rel_lap
                            ):
                                best_dcim_rel_lap = float(rel_lap)
                            if math.isfinite(float(score_mid)) and (
                                best_dcim_score is None or float(score_mid) < best_dcim_score
                            ):
                                best_dcim_score = float(score_mid)
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
                    if payload is not None:
                        per_payload = payload.for_program(idx)
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
                    A_train_scaled, col_norms = _scale_columns(A_train)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    weights_scaled, _ = solve_sparse(
                        A_train_scaled,
                        X_train,
                        V_train,
                        is_boundary,
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
                    end.record()
                    torch.cuda.synchronize()
                    t_solve_ms = float(start.elapsed_time(end))
                    system = ImageSystem(elements, weights)

                    if cache_hold_matrices and idx in hold_cache:
                        A_hold = hold_cache.pop(idx)
                    else:
                        A_hold = assemble_basis_matrix(elements, X_hold)
                    assert_cuda_tensor(A_hold, "A_hold_fit")
                    pred_hold = A_hold.matmul(weights)
                    bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                    in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                    mean_bc = float(torch.mean(bc_err).item()) if bc_err.numel() else 0.0
                    mean_in = float(torch.mean(in_err).item()) if in_err.numel() else 0.0
                    max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                    max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                    mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                    mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                    mean_bc_mid = float(torch.mean(mid_bc_err).item()) if mid_bc_err.numel() else 0.0
                    mean_in_mid = float(torch.mean(mid_in_err).item()) if mid_in_err.numel() else 0.0
                    rel_bc = mean_bc_mid / max(oracle_bc_mean_abs, 1e-12)
                    rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                    def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                        assert_cuda_tensor(pts, "candidate_eval_points")
                        out = system.potential(pts)
                        assert_cuda_tensor(out, "candidate_eval_out")
                        return out

                    lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                    lap_abs = torch.abs(lap)
                    lap_mean = float(torch.mean(lap_abs).item()) if lap_abs.numel() else 0.0
                    lap_max = float(torch.max(lap_abs).item()) if lap_abs.numel() else 0.0
                    rel_lap = lap_mean / lap_denom

                    eval_ms = _timed_cuda(_eval_fn, interior_hold, warmup=1, repeat=3)

                    perturbed = _perturb_elements(elements, stability_sigma, device=device)
                    stability_ratio = float("nan")
                    if perturbed is not None:
                        A_hold_pert = assemble_basis_matrix(perturbed, X_hold)
                        assert_cuda_tensor(A_hold_pert, "A_hold_pert")
                        pert_pred = A_hold_pert.matmul(weights)
                        base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                        pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                        pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                        pert_err = w_bc * float(torch.mean(pert_bc).item()) + w_pde * float(torch.mean(pert_in).item())
                        stability_ratio = float(pert_err / max(base_err, 1e-8))

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
                        "eval_time_us": float(eval_ms * 1000.0) if eval_ms == eval_ms else float("nan"),
                        "solve_time_us": float(t_solve_ms * 1000.0) if t_solve_ms == t_solve_ms else float("nan"),
                        "total_time_us": float((eval_ms + t_solve_ms) * 1000.0) if eval_ms == eval_ms else float("nan"),
                        "complexity_terms": comp["n_terms"],
                        "complexity_nodes": comp["n_nodes"],
                        "n_terms": comp["n_terms"],
                        "complex_count": int(complex_count),
                    }
                    score_mid = (
                        w_bc * rel_bc
                        + w_pde * rel_lap
                        + w_complexity * comp["n_terms"]
                        + w_latency * (metrics["eval_time_us"] / 1e6)
                        + w_stability * stability_ratio
                    )
                    fitted_candidates.append(
                        {
                            "program": program,
                            "elements": elements,
                            "weights": weights,
                            "metrics": metrics,
                            "score": float(score_mid),
                        }
                    )

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
                        pred_hi = A_hold_hi.matmul(cand["weights"])
                        bc_err_hi = torch.mean(torch.abs(pred_hi[is_boundary_hold] - V_hold_hi[is_boundary_hold])).item()
                        in_err_hi = torch.mean(torch.abs(pred_hi[~is_boundary_hold] - V_hold_hi[~is_boundary_hold])).item()
                        cand["score_hi"] = float(w_bc * bc_err_hi + w_pde * in_err_hi)
                        cand["metrics"]["hi_bc_mean_abs"] = float(bc_err_hi)
                        cand["metrics"]["hi_pde_mean_abs"] = float(in_err_hi)
                    top_final = sorted(top_mid, key=lambda x: x.get("score_hi", x["score"]))[
                        : max(1, min(topk_final, len(top_mid)))
                    ]
                else:
                    top_final = top_mid[: max(1, min(topk_final, len(top_mid)))]

                if refine_enabled and is_layered and top_final:
                    if refine_targets == "holdout":
                        refine_bc = bc_hold
                        refine_interior = interior_hold
                        refine_bc_target = V_bc_hold_mid if V_bc_hold_mid is not None else V_bc_hold
                        refine_X = X_hold
                        refine_V = V_hold_mid
                        refine_is_boundary = is_boundary_hold
                    else:
                        refine_bc = bc_train
                        refine_interior = interior_train
                        refine_bc_target = V_bc_train
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
                        A_refine_scaled, col_norms = _scale_columns(A_refine)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        weights_scaled, _ = solve_sparse(
                            A_refine_scaled,
                            refine_X,
                            refine_V,
                            refine_is_boundary,
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
                        end.record()
                        torch.cuda.synchronize()
                        t_solve_ms = float(start.elapsed_time(end))

                        system_refined = ImageSystem(refine_elements, refined_weights)

                        def _eval_fn(pts: torch.Tensor) -> torch.Tensor:
                            assert_cuda_tensor(pts, "refined_eval_points")
                            with torch.no_grad():
                                out = system_refined.potential(pts)
                            assert_cuda_tensor(out, "refined_eval_out")
                            return out

                        with torch.no_grad():
                            A_hold = assemble_basis_matrix(refine_elements, X_hold)
                            assert_cuda_tensor(A_hold, "A_hold_refined")
                            pred_hold = A_hold.matmul(refined_weights)
                            bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold[is_boundary_hold])
                            in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold[~is_boundary_hold])
                            mean_bc = float(torch.mean(bc_err).item()) if bc_err.numel() else 0.0
                            mean_in = float(torch.mean(in_err).item()) if in_err.numel() else 0.0
                            max_bc = float(torch.max(bc_err).item()) if bc_err.numel() else 0.0
                            max_in = float(torch.max(in_err).item()) if in_err.numel() else 0.0

                            mid_bc_err = torch.abs(pred_hold[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            mid_in_err = torch.abs(pred_hold[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            mean_bc_mid = float(torch.mean(mid_bc_err).item()) if mid_bc_err.numel() else 0.0
                            mean_in_mid = float(torch.mean(mid_in_err).item()) if mid_in_err.numel() else 0.0
                            rel_bc = mean_bc_mid / denom_bc
                            rel_in = mean_in_mid / max(oracle_in_mean_abs, 1e-12)

                            lap = _laplacian_fd(_eval_fn, interior_hold, h=1e-2)
                            lap_abs = torch.abs(lap)
                            lap_mean = float(torch.mean(lap_abs).item()) if lap_abs.numel() else 0.0
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
                            pert_pred = A_hold_pert.matmul(refined_weights)
                            base_err = w_bc * mean_bc_mid + w_pde * mean_in_mid
                            pert_bc = torch.abs(pert_pred[is_boundary_hold] - V_hold_mid[is_boundary_hold])
                            pert_in = torch.abs(pert_pred[~is_boundary_hold] - V_hold_mid[~is_boundary_hold])
                            pert_err = w_bc * float(torch.mean(pert_bc).item()) + w_pde * float(torch.mean(pert_in).item())
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
                    system = ImageSystem(cand["elements"], cand["weights"])

                    def _verify_eval(pts: torch.Tensor) -> torch.Tensor:
                        assert_cuda_tensor(pts, "verifier_points")
                        out = system.potential(pts)
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
                    mean_bc = float(cand["metrics"].get("mean_bc_err_holdout", float("inf")))
                    if best_mean_bc is None or mean_bc < best_mean_bc:
                        best_mean_bc = mean_bc
                    rel_bc_val = float(cand["metrics"].get("rel_bc_err_holdout", float("inf")))
                    if math.isfinite(rel_bc_val) and (best_rel_bc is None or rel_bc_val < best_rel_bc):
                        best_rel_bc = rel_bc_val
                        best_rel_in = float(cand["metrics"].get("rel_pde_err_holdout", float("inf")))
                        best_rel_elements = cand["elements"]
                        best_rel_metrics = cand["metrics"]
                    rel_in_val = float(cand["metrics"].get("rel_pde_err_holdout", float("inf")))
                    if math.isfinite(rel_in_val) and (best_rel_in is None or rel_in_val < best_rel_in):
                        best_rel_in = rel_in_val
                        best_rel_in_metrics = cand["metrics"]
                    rel_lap_val = float(cand["metrics"].get("rel_lap_holdout", float("inf")))
                    if math.isfinite(rel_lap_val) and (best_rel_lap is None or rel_lap_val < best_rel_lap):
                        best_rel_lap = rel_lap_val
                        best_rel_lap_metrics = cand["metrics"]

                    cert_dir = run_dir / "artifacts" / "certificates" / f"gen{gen:03d}_rank{rank}_verifier"
                    cert_status = "error"
                    cert_error = None
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

                gen_rel_bc = float("inf")
                gen_rel_in = float("inf")
                gen_rel_lap = float("inf")
                for cand in top_final:
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
                        (ramp_best_rel_bc is not None or ramp_best_rel_lap is not None)
                        and gen - last_improve_gen >= ramp_patience
                    ):
                        print("RAMP ABORT: not improving")
                        ramp_abort = True
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
            if ramp_abort:
                break
        if ramp_abort:
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
        print(
            "RUN SUMMARY:"
            f" best_bc_abs={bc_abs:.3e}"
            f" best_bc_rel={bc_rel:.3e}"
            f" best_in_abs={in_abs:.3e}"
            f" best_in_rel={in_rel:.3e}"
            f" best_lap_abs={lap_abs:.3e}"
            f" best_lap_rel={lap_rel:.3e}"
        )
    else:
        print("RUN SUMMARY: no candidates evaluated")

    print(f"RUN DIR: {run_dir}")
    print(f"ENV JSON: {run_dir / 'env.json'}")

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
        print("TOP 5 CANDIDATES (by rel_bc):")
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
            print(
                f"{idx:02d} mean_bc={mean_bc:.3e} rel_bc={rel_bc:.3e} "
                f"mean_in={mean_in:.3e} rel_in={rel_in:.3e} rel_lap={rel_lap:.3e} "
                f"complex_count={complex_count} n_terms={n_terms}{dcim_marker}"
            )
    else:
        print("TOP 5 CANDIDATES: none")

    if best_rel_elements:
        hist = _element_type_hist(best_rel_elements)
        hist_str = ", ".join(f"{k}: {v}" for k, v in hist.items())
        dcim_present = bool(best_rel_metrics and best_rel_metrics.get("is_dcim_block_baseline"))
        dcim_present = dcim_present or _has_dcim_block(best_rel_elements)
        print(f"BEST ELEMENT TYPES: {hist_str} | dcim_block_present={dcim_present}")
    else:
        print("BEST ELEMENT TYPES: none")

    dcim_used_as_best = False
    if best_rel_metrics is not None and best_rel_metrics.get("is_dcim_block_baseline"):
        dcim_used_as_best = True
    if best_rel_elements and _has_dcim_block(best_rel_elements):
        dcim_used_as_best = True
    dcim_rel_bc = best_dcim_rel_bc if best_dcim_rel_bc is not None else float("nan")
    dcim_rel_lap = best_dcim_rel_lap if best_dcim_rel_lap is not None else float("nan")
    dcim_score = best_dcim_score if best_dcim_score is not None else float("nan")
    print(
        f"DCIM BASELINE BEST: rel_bc={dcim_rel_bc:.3e} rel_lap={dcim_rel_lap:.3e} "
        f"score={dcim_score:.3e} used_as_best={dcim_used_as_best}"
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
        print(f"SPEC HASH: {spec_hash_report} ({status})")
    else:
        print(f"SPEC HASH: {spec_hash_report}")
    print(
        "RAMP SIGNAL:"
        f" improved_bc={improved_bc}"
        f" improved_lap={improved_lap}"
        f" best_bc_rel={best_bc_rel:.3e}"
        f" best_lap_rel={best_lap_rel:.3e}"
        f" ready={ready}"
    )
    if ready:
        print("READY for monster run")
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
        print(f"NOT READY: {limiter}")
    if refine_enabled:
        print(f"REFINE IMPROVED: {refine_improved_total}/{refine_attempted_total}")
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
            print(f"REFINE BEST REL: rel_bc={ref_bc:.3e} rel_lap={ref_lap:.3e}")

    if ramp_abort:
        return 3

    if sanity_threshold is not None:
        if best_mean_bc is None or best_mean_bc > sanity_threshold:
            print(
                f"Sanity FAIL: best mean BC holdout {best_mean_bc} exceeds threshold {sanity_threshold}."
            )
            return 2
        print(
            f"Sanity PASS: best mean BC holdout {best_mean_bc} within threshold {sanity_threshold}."
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
