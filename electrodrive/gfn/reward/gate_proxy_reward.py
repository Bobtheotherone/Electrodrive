"""Gate proxy reward for GFlowNet training aligned to verifier gates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import math

import numpy as np
import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.flows.types import ParamPayload
from electrodrive.gfn.dsl.nodes import AddBranchCutBlock, AddPoleBlock, AddPrimitiveBlock
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.gfn.reward.reward import RewardTerms, clip_log_reward
from electrodrive.images.basis import DCIMBranchCutImageBasis, DCIMPoleImageBasis, ImageBasisElement
from electrodrive.images.basis_dcim import DCIMBlockBasis
from electrodrive.images.search import (
    ImageSystem,
    assemble_basis_matrix,
    _prepare_collocation_batch,
    _resolve_collocation_params,
)
from electrodrive.learn.collocation import compute_layered_reference_potential, make_collocation_batch_for_spec
from electrodrive.utils.device import get_default_device
from electrodrive.verify.gate_proxies import proxy_gateA, proxy_gateB, proxy_gateC, proxy_gateD


@dataclass(frozen=True)
class GateProxyRewardWeights:
    """Weights for combining proxy metrics into a log-reward."""

    gateA: float = 1.0
    gateB: float = 1.0
    gateC: float = 1.0
    gateD: float = 1.0
    speed: float = 0.05
    complexity: float = 0.02
    dcim_bonus: float = 0.1
    complex_bonus: float = 0.1


@dataclass
class GateProxyRewardConfig:
    """Configuration for gate-proxy reward computation."""

    n_points: Optional[int] = None
    ratio_boundary: Optional[float] = None
    supervision_mode: str = "auto"
    reg_l2: float = 1e-3
    dtype: torch.dtype = torch.float32
    logR_clip: Tuple[float, float] = (-20.0, 20.0)
    nonfinite_penalty: float = 1e4
    max_term: float = 1e6
    collocation_cache_size: int = 4
    weights: GateProxyRewardWeights = field(default_factory=GateProxyRewardWeights)

    gateA_n_interior: int = 64
    gateA_exclusion_radius: float = 5e-2
    gateA_interface_band: float = 1e-2
    gateA_fd_h: float = 2e-2
    gateA_prefer_autograd: bool = False
    gateA_autograd_max_samples: Optional[int] = None
    gateA_fd_max_samples: int = 128
    gateA_transform: str = "logcap"
    gateA_cap: float = 1e6
    gateA_linf_tol: float = 5e-3

    gateB_n_xy: int = 64
    interface_delta: float = 1e-2

    gateC_n_dir: int = 64
    near_radii: Tuple[float, float] = (0.125, 0.5)
    far_radii: Tuple[float, float] = (10.0, 20.0)

    gateD_n_points: int = 64
    stability_delta: float = 1e-2

    use_reference_potential: bool = True
    param_fallback: bool = True
    fallback_latent_dim: int = 4
    fallback_schema_id: int = SCHEMA_COMPLEX_DEPTH

    dcim_term_cost: float = 1.5
    dcim_block_cost: float = 2.5


class GateProxyRewardComputer:
    """Reward computer that scores programs via fast verifier proxy metrics."""

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        config: Optional[GateProxyRewardConfig] = None,
    ) -> None:
        self.device = device or get_default_device()
        self.config = config or GateProxyRewardConfig()
        self.param_sampler = None
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_diagnostics_batch: list[Dict[str, Any]] = []
        self._colloc_cache: OrderedDict[Tuple[str, int, int, float, str], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = OrderedDict()

    def compute(
        self,
        program: Program,
        spec: object,
        device: Optional[torch.device] = None,
        *,
        seed: Optional[int] = None,
        spec_embedding: Optional[torch.Tensor] = None,
        param_payload: Optional[ParamPayload] = None,
    ) -> RewardTerms:
        _ = spec_embedding
        cfg = self.config
        device = device or self.device
        dtype = cfg.dtype
        self.last_diagnostics = {}

        if param_payload is None and cfg.param_fallback and device.type == "cuda":
            param_payload = _fallback_payload(
                program,
                device=device,
                dtype=dtype,
                seed=seed,
                latent_dim=cfg.fallback_latent_dim,
                schema_id=cfg.fallback_schema_id,
            )
            self.last_diagnostics["param_fallback"] = True

        try:
            elements, _, _ = compile_program_to_basis(
                program,
                spec,
                device,
                param_payload=param_payload,
                strict=False,
            )
        except Exception as exc:
            self.last_diagnostics["compile_error"] = str(exc)
            return _empty_reward(program, device, dtype, cfg)

        if not elements:
            self.last_diagnostics["empty_compilation"] = True
            return _empty_reward(program, device, dtype, cfg)

        X_train, V_train, _ = self._get_collocation(spec, device, dtype, cfg, seed=seed)
        if X_train.numel() == 0 or V_train.numel() == 0:
            self.last_diagnostics["solver_failed"] = True
            return _empty_reward(program, device, dtype, cfg)

        A_train = assemble_basis_matrix(elements, X_train)
        weights = _fast_weights(A_train, V_train, reg=cfg.reg_l2, normalize=True)
        if weights.numel() == 0 or not torch.isfinite(weights).all():
            self.last_diagnostics["nonfinite"] = True
            return _empty_reward(program, device, dtype, cfg)

        system = ImageSystem(elements, weights)
        use_ref = bool(cfg.use_reference_potential and getattr(spec, "BCs", "") == "dielectric_interfaces")

        def _candidate_eval(pts: torch.Tensor) -> torch.Tensor:
            out = system.potential(pts)
            if use_ref:
                out = out + compute_layered_reference_potential(
                    spec,
                    pts,
                    device=pts.device,
                    dtype=pts.dtype,
                )
            return out

        try:
            proxy_metrics = self._compute_proxy_metrics(
                spec,
                _candidate_eval,
                cfg,
                device=device,
                seed=seed,
                program=program,
            )
        except Exception as exc:
            self.last_diagnostics["proxy_error"] = str(exc)
            return _empty_reward(program, device, dtype, cfg)
        a_term = _proxyA_term(proxy_metrics, cfg)
        b_term = _proxyB_term(proxy_metrics, cfg)
        c_term = _proxyC_term(proxy_metrics, cfg)
        d_term = _proxyD_term(proxy_metrics, cfg)

        dcim_bonus = float(cfg.weights.dcim_bonus) if _program_has_dcim(program) else 0.0
        complex_bonus = float(cfg.weights.complex_bonus) if _program_has_complex(program) else 0.0
        bonus = dcim_bonus + complex_bonus

        dcim_stats = _dcim_stats(elements)
        n_terms = int(len(elements))
        speed_proxy = _speed_proxy(
            n_terms,
            dcim_stats.get("dcim_terms", 0),
            dcim_stats.get("dcim_blocks", 0),
            dcim_term_cost=cfg.dcim_term_cost,
            dcim_block_cost=cfg.dcim_block_cost,
        )
        complexity = float(len(program.nodes))

        w = cfg.weights
        score = (
            w.gateA * a_term
            + w.gateB * b_term
            + w.gateC * c_term
            + w.gateD * d_term
            + w.speed * speed_proxy
            + w.complexity * complexity
        )
        logR = torch.tensor(bonus - score, device=device, dtype=dtype)
        logR = clip_log_reward(logR, cfg.logR_clip[0], cfg.logR_clip[1])
        logR = torch.nan_to_num(logR, nan=cfg.logR_clip[0], posinf=cfg.logR_clip[1], neginf=cfg.logR_clip[0])

        self.last_diagnostics.update(
            {
                "proxy_metrics": dict(proxy_metrics),
                "proxy_score": float(score),
                "speed_proxy": float(speed_proxy),
                "dcim_bonus": float(dcim_bonus),
                "complex_bonus": float(complex_bonus),
                "n_terms": int(n_terms),
                "n_nodes": int(len(program.nodes)),
            }
        )

        return RewardTerms(
            relerr=torch.tensor(a_term, device=device, dtype=dtype),
            latency_ms=torch.tensor(speed_proxy, device=device, dtype=dtype),
            instability=torch.tensor(b_term + c_term + d_term, device=device, dtype=dtype),
            complexity=torch.tensor(complexity, device=device, dtype=dtype),
            novelty=torch.tensor(bonus, device=device, dtype=dtype),
            logR=logR,
        )

    def _get_collocation(
        self,
        spec: object,
        device: torch.device,
        dtype: torch.dtype,
        cfg: GateProxyRewardConfig,
        *,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        n_points, ratio_boundary = _resolve_collocation_params(cfg.n_points, cfg.ratio_boundary)
        spec_hash = _infer_spec_hash(spec)
        cache_key = (spec_hash, int(seed or 0), int(n_points), float(ratio_boundary), cfg.supervision_mode)
        if cache_key in self._colloc_cache:
            X, V, is_boundary = self._colloc_cache[cache_key]
            self._colloc_cache.move_to_end(cache_key)
            return X, V, is_boundary

        rng = np.random.default_rng(int(seed) if seed is not None else None)
        with torch.no_grad():
            batch = make_collocation_batch_for_spec(
                spec=spec,
                n_points=n_points,
                ratio_boundary=ratio_boundary,
                supervision_mode=cfg.supervision_mode,
                device=device,
                dtype=dtype,
                rng=rng,
            )
        X_f, V_f, is_boundary_out, _ = _prepare_collocation_batch(
            batch,
            device=device,
            dtype=dtype,
            logger=_NullLogger(),
            return_is_boundary=True,
            pass_label="gate_proxy_reward",
        )
        self._colloc_cache[cache_key] = (X_f, V_f, is_boundary_out)
        if len(self._colloc_cache) > cfg.collocation_cache_size:
            self._colloc_cache.popitem(last=False)
        return X_f, V_f, is_boundary_out

    def _compute_proxy_metrics(
        self,
        spec: object,
        candidate_eval: Any,
        cfg: GateProxyRewardConfig,
        *,
        device: torch.device,
        seed: Optional[int],
        program: Optional[Program] = None,
    ) -> Dict[str, Any]:
        _ = program
        seed_val = int(seed or 0)
        metrics: Dict[str, Any] = {}
        metrics.update(
            proxy_gateA(
                spec,
                candidate_eval,
                n_interior=int(cfg.gateA_n_interior),
                exclusion_radius=float(cfg.gateA_exclusion_radius),
                fd_h=float(cfg.gateA_fd_h),
                prefer_autograd=bool(cfg.gateA_prefer_autograd),
                interface_band=float(cfg.gateA_interface_band),
                device=device,
                dtype=torch.float32,
                seed=seed_val + 1,
                linf_tol=float(cfg.gateA_linf_tol),
                autograd_max_samples=cfg.gateA_autograd_max_samples,
                fd_max_samples=int(cfg.gateA_fd_max_samples),
            )
        )
        metrics.update(
            proxy_gateB(
                spec,
                candidate_eval,
                n_xy=int(cfg.gateB_n_xy),
                delta=float(cfg.interface_delta),
                device=device,
                dtype=torch.float32,
                seed=seed_val + 3,
            )
        )
        metrics.update(
            proxy_gateC(
                candidate_eval,
                near_radii=cfg.near_radii,
                far_radii=cfg.far_radii,
                n_dir=int(cfg.gateC_n_dir),
                device=device,
                dtype=torch.float32,
                seed=seed_val + 5,
            )
        )
        proxy_pts = _proxy_stability_points(spec, device=device, dtype=torch.float32, n_points=int(cfg.gateD_n_points))
        metrics.update(
            proxy_gateD(
                candidate_eval,
                proxy_pts,
                delta=float(cfg.stability_delta),
                seed=seed_val + 7,
            )
        )
        return metrics


class _NullLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


def _infer_spec_hash(spec: object) -> str:
    if isinstance(spec, str):
        return spec
    for attr in ("spec_hash", "hash", "id"):
        if hasattr(spec, attr):
            return str(getattr(spec, attr))
    return str(spec)


def _fallback_payload(
    program: Program,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int],
    latent_dim: int,
    schema_id: int,
) -> ParamPayload:
    mapping: list[int] = []
    schema_ids: list[int] = []
    token_idx = 0
    for node in program.nodes:
        if isinstance(node, (AddPrimitiveBlock, AddPoleBlock, AddBranchCutBlock)):
            mapping.append(token_idx)
            node_schema = getattr(node, "schema_id", None)
            schema_ids.append(int(node_schema) if node_schema is not None else int(schema_id))
            token_idx += 1
        else:
            mapping.append(-1)
    if token_idx == 0:
        token_idx = 1
        schema_ids = [int(schema_id)]
        mapping = [0]

    u_latent = torch.zeros((token_idx, int(latent_dim)), device=device, dtype=dtype)
    node_mask = torch.ones((token_idx,), device=device, dtype=torch.bool)
    return ParamPayload(
        u_latent=u_latent,
        node_mask=node_mask,
        dim_mask=None,
        schema_ids=torch.tensor(schema_ids, device=device, dtype=torch.long),
        node_to_token=mapping,
        seed=int(seed) if seed is not None else None,
        config_hash="gate_proxy_fallback",
        device=device,
        dtype=dtype,
    )


def _proxy_stability_points(
    spec: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int,
) -> torch.Tensor:
    n = max(1, int(n_points))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        batch = make_collocation_batch_for_spec(
            spec=spec,
            n_points=n,
            ratio_boundary=0.5,
            supervision_mode="auto",
            device=device,
            dtype=dtype,
            rng=rng,
        )
    X = batch.get("X")
    if X is None or X.numel() == 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    return X[:n].contiguous()


def _proxyA_term(metrics: Mapping[str, Any], cfg: GateProxyRewardConfig) -> float:
    a_raw = _safe_metric(metrics.get("proxy_gateA_worst_ratio", cfg.gateA_cap), cfg)
    a_clamped = min(a_raw, float(cfg.gateA_cap))
    if str(cfg.gateA_transform).lower() == "logcap":
        return math.log10(1.0 + a_clamped)
    return a_clamped


def _proxyB_term(metrics: Mapping[str, Any], cfg: GateProxyRewardConfig) -> float:
    b_v = _safe_metric(metrics.get("proxy_gateB_max_v_jump", 0.0), cfg)
    b_d = _safe_metric(metrics.get("proxy_gateB_max_d_jump", 0.0), cfg)
    return max(b_v, b_d)


def _proxyC_term(metrics: Mapping[str, Any], cfg: GateProxyRewardConfig) -> float:
    far_slope = _safe_metric(metrics.get("proxy_gateC_far_slope", 0.0), cfg)
    near_slope = _safe_metric(metrics.get("proxy_gateC_near_slope", 0.0), cfg)
    spurious = _safe_metric(metrics.get("proxy_gateC_spurious_fraction", 0.0), cfg)
    return abs(far_slope + 1.0) + abs(near_slope + 1.0) + 10.0 * spurious


def _proxyD_term(metrics: Mapping[str, Any], cfg: GateProxyRewardConfig) -> float:
    return _safe_metric(metrics.get("proxy_gateD_rel_change", 0.0), cfg)


def _safe_metric(value: Any, cfg: GateProxyRewardConfig) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(cfg.nonfinite_penalty)
    if not math.isfinite(val):
        return float(cfg.nonfinite_penalty)
    if val < 0.0:
        return 0.0
    return float(min(val, cfg.max_term))


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


def _program_has_dcim(program: Program) -> bool:
    return any(isinstance(node, (AddPoleBlock, AddBranchCutBlock)) for node in program.nodes)


def _program_has_complex(program: Program) -> bool:
    for node in program.nodes:
        if isinstance(node, AddPrimitiveBlock):
            try:
                if int(node.schema_id or 0) == SCHEMA_COMPLEX_DEPTH:
                    return True
            except Exception:
                continue
    return False


def _scale_columns(A: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.numel() == 0:
        return A, torch.ones((A.shape[1],), device=A.device, dtype=A.dtype)
    col_norms = torch.linalg.norm(A, dim=0).clamp_min(eps)
    return A / col_norms, col_norms


def _fast_weights(
    A: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    if A.numel() == 0 or A.shape[1] == 0:
        return torch.zeros((0,), device=A.device, dtype=A.dtype)
    if normalize:
        A_scaled, col_norms = _scale_columns(A)
    else:
        A_scaled = A
        col_norms = torch.ones((A.shape[1],), device=A.device, dtype=A.dtype)
    reg_val = float(reg) if math.isfinite(float(reg)) else 0.0
    ata = A_scaled.transpose(0, 1).matmul(A_scaled)
    ata = ata + reg_val * torch.eye(A_scaled.shape[1], device=A.device, dtype=A.dtype)
    atb = A_scaled.transpose(0, 1).matmul(b)
    try:
        w_scaled = torch.linalg.solve(ata, atb)
    except RuntimeError:
        try:
            w_scaled = torch.linalg.lstsq(A_scaled, b).solution
        except RuntimeError:
            return torch.zeros((A.shape[1],), device=A.device, dtype=A.dtype)
    return w_scaled / col_norms


def _empty_reward(
    program: Program,
    device: torch.device,
    dtype: torch.dtype,
    cfg: GateProxyRewardConfig,
) -> RewardTerms:
    relerr = torch.tensor(cfg.nonfinite_penalty, device=device, dtype=dtype)
    instability = torch.tensor(cfg.nonfinite_penalty, device=device, dtype=dtype)
    complexity = torch.tensor(float(len(getattr(program, "nodes", []) or [])), device=device, dtype=dtype)
    novelty = torch.tensor(0.0, device=device, dtype=dtype)
    logR = torch.tensor(cfg.logR_clip[0], device=device, dtype=dtype)
    latency = torch.tensor(0.0, device=device, dtype=dtype)
    return RewardTerms(
        relerr=relerr,
        latency_ms=latency,
        instability=instability,
        complexity=complexity,
        novelty=novelty,
        logR=logR,
    )


__all__ = [
    "GateProxyRewardComputer",
    "GateProxyRewardConfig",
    "GateProxyRewardWeights",
]
