"""Reward interfaces and normalization utilities for GFlowNet training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from collections import OrderedDict
import time

import numpy as np
import torch

from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.integration.compile import compile_program_to_basis
from electrodrive.images.basis import ImageBasisElement
from electrodrive.images.search import ImageSystem, assemble_basis, solve_sparse, _prepare_collocation_batch, _resolve_collocation_params
from electrodrive.learn.collocation import make_collocation_batch_for_spec
from electrodrive.discovery import novelty as novelty_lib
from electrodrive.images.structural_features import structural_fingerprint
from electrodrive.utils.device import get_default_device


@dataclass(frozen=True)
class RewardTerms:
    """Decomposed reward components used by GFlowNet losses."""

    relerr: torch.Tensor
    latency_ms: torch.Tensor
    instability: torch.Tensor
    complexity: torch.Tensor
    novelty: torch.Tensor
    logR: torch.Tensor

    def as_dict(self) -> Mapping[str, torch.Tensor]:
        """Return reward terms as a mapping."""
        return {
            "relerr": self.relerr,
            "latency_ms": self.latency_ms,
            "instability": self.instability,
            "complexity": self.complexity,
            "novelty": self.novelty,
            "logR": self.logR,
        }

    def to(self, device: torch.device) -> "RewardTerms":
        """Move reward tensors to the specified device."""
        return RewardTerms(
            relerr=self.relerr.to(device),
            latency_ms=self.latency_ms.to(device),
            instability=self.instability.to(device),
            complexity=self.complexity.to(device),
            novelty=self.novelty.to(device),
            logR=self.logR.to(device),
        )

    def detach_cpu(self) -> "RewardTerms":
        """Detach reward tensors and move them to CPU for replay/storage."""
        return RewardTerms(
            relerr=self.relerr.detach().to("cpu"),
            latency_ms=self.latency_ms.detach().to("cpu"),
            instability=self.instability.detach().to("cpu"),
            complexity=self.complexity.detach().to("cpu"),
            novelty=self.novelty.detach().to("cpu"),
            logR=self.logR.detach().to("cpu"),
        )


@dataclass
class RewardNormalizer:
    """Running normalization statistics for log-rewards."""

    device: torch.device = get_default_device()
    eps: float = 1e-6
    mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    var: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    count: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def __post_init__(self) -> None:
        self.mean = self.mean.to(self.device)
        self.var = self.var.to(self.device)
        self.count = self.count.to(self.device)

    def update(self, logR: torch.Tensor) -> None:
        """Update running mean/variance using a batch of log-rewards."""
        if logR.numel() == 0:
            return
        with torch.no_grad():
            batch_mean = logR.mean()
            batch_var = logR.var(unbiased=False)
            batch_count = torch.tensor(float(logR.numel()), device=self.device)
            total = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta * delta * self.count * batch_count / total
            self.mean = new_mean
            self.var = m2 / total
            self.count = total

    def normalize(self, logR: torch.Tensor) -> torch.Tensor:
        """Normalize log-rewards using running statistics."""
        return (logR - self.mean) / torch.sqrt(self.var + self.eps)


@dataclass(frozen=True)
class RewardWeights:
    """Weights for combining reward components into logR."""

    alpha: float = 1.0
    beta: float = 1e-3
    gamma: float = 1.0
    delta: float = 0.1
    eta: float = 0.2


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    n_points: Optional[int] = None
    ratio_boundary: Optional[float] = None
    supervision_mode: str = "auto"
    operator_mode: bool = True
    solver: str = "ista"
    reg_l1: float = 1e-3
    max_iter: int = 200
    tol: float = 1e-6
    boundary_weight: Optional[float] = None
    lambda_group: float = 0.0
    logR_clip: Tuple[float, float] = (-20.0, 20.0)
    weights: RewardWeights = field(default_factory=RewardWeights)
    latency_warmup: int = 1
    cond_threshold: float = 1e7
    sensitivity_weight: float = 1.0
    weight_range_weight: float = 0.1
    nonfinite_penalty: float = 5.0
    complexity_weights: Tuple[float, float, float] = (0.5, 1.0, 0.2)
    collocation_cache_size: int = 4
    compile_cache_size: int = 128
    cache_on_gpu: bool = False
    dtype: torch.dtype = torch.float32


class _NullLogger:
    """No-op logger for reward computations."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass


class RewardComputer:
    """GPU-first reward computer based on the existing solver stack."""

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        config: Optional[RewardConfig] = None,
    ) -> None:
        self.device = device or get_default_device()
        self.config = config or RewardConfig()
        self._compile_cache: OrderedDict[Tuple[str, str, str], Tuple[Any, torch.Tensor, Mapping[str, Any]]] = OrderedDict()
        self._colloc_cache: OrderedDict[Tuple[str, int, int, float, str], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = OrderedDict()
        self.last_diagnostics: Dict[str, Any] = {}

    def compute(
        self,
        program: Program,
        spec: object,
        device: Optional[torch.device] = None,
        *,
        seed: Optional[int] = None,
        config: Optional[RewardConfig] = None,
    ) -> RewardTerms:
        """Compute reward terms for a program/spec pair."""
        cfg = config or self.config
        device = device or self.device
        dtype = cfg.dtype
        self.last_diagnostics = {}

        def _run() -> RewardTerms:
            timer = _LatencyTimer(device)
            timer.start()

            elements, group_ids, meta = self._compile_program(program, spec, device, dtype, cfg)
            if not elements:
                timer.stop()
                return self._empty_reward(program, device, dtype, cfg, timer.elapsed_ms())

            colloc_seed = seed if seed is not None else None
            holdout_seed = (seed + 1) if seed is not None else None
            X_train, V_train, is_boundary = self._get_collocation(
                spec,
                device,
                dtype,
                cfg,
                seed=colloc_seed,
            )
            X_hold, V_hold, _ = self._get_collocation(
                spec,
                device,
                dtype,
                cfg,
                seed=holdout_seed,
            )

            if X_train.numel() == 0 or V_train.numel() == 0:
                timer.stop()
                self.last_diagnostics["solver_failed"] = True
                return self._empty_reward(program, device, dtype, cfg, timer.elapsed_ms())

            try:
                A = assemble_basis(elements, X_train, cfg.operator_mode, device, dtype)
            except Exception:
                A = assemble_basis(elements, X_train, False, device, dtype)

            if device.type == "cuda" and cfg.latency_warmup > 0:
                _warmup_matvec(A, X_train, device, dtype)

            weights, _, stats = solve_sparse(
                A,
                X_train,
                V_train,
                is_boundary,
                _NullLogger(),
                reg_l1=cfg.reg_l1,
                solver=cfg.solver,
                lista_model=None,
                aug_lagrange_cfg=None,
                boundary_weight=cfg.boundary_weight,
                boundary_mode="penalty",
                per_elem_reg=None,
                group_ids=group_ids,
                lambda_group=cfg.lambda_group,
                weight_prior=None,
                lambda_weight_prior=0.0,
                normalize_columns=True,
                ls_refit=False,
                support_k=None,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                lista_refine=False,
                return_stats=True,
            )

            if not torch.isfinite(weights).all():
                self.last_diagnostics["nonfinite"] = True

            pred_train = _matvec(A, weights, X_train)
            pred_hold = _matvec(A, weights, X_hold) if X_hold.numel() else pred_train

            relerr_train = _rel_resid(pred_train, V_train)
            relerr_hold = _rel_resid(pred_hold, V_hold) if V_hold.numel() else relerr_train
            relerr = torch.maximum(relerr_train, relerr_hold)

            cond_penalty = _condition_penalty(A, X_train, device, dtype, cfg)
            sensitivity = torch.abs(relerr_hold - relerr_train)
            range_penalty = _weight_range_penalty(weights)

            instability = (
                cfg.sensitivity_weight * sensitivity
                + cond_penalty
                + cfg.weight_range_weight * range_penalty
            )
            if self.last_diagnostics.get("nonfinite"):
                instability = instability + torch.tensor(cfg.nonfinite_penalty, device=device, dtype=dtype)

            complexity = _complexity_score(program, elements, meta, device, dtype, cfg)
            novelty = _novelty_score(elements, weights, spec, device, dtype)
            timer.stop()
            latency_ms = timer.elapsed_ms_tensor(device, dtype)
            logR = _combine_log_reward(relerr, latency_ms, instability, complexity, novelty, cfg)
            logR = clip_log_reward(logR, cfg.logR_clip[0], cfg.logR_clip[1])
            logR = torch.nan_to_num(logR, nan=cfg.logR_clip[0], posinf=cfg.logR_clip[1], neginf=cfg.logR_clip[0])

            return RewardTerms(
                relerr=relerr,
                latency_ms=latency_ms,
                instability=instability,
                complexity=complexity,
                novelty=novelty,
                logR=logR,
            )

        if seed is None:
            return _run()

        devices = []
        if device.type == "cuda":
            devices = [device.index] if device.index is not None else [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            return _run()

    def _compile_program(
        self,
        program: Program,
        spec: object,
        device: torch.device,
        dtype: torch.dtype,
        cfg: RewardConfig,
    ) -> Tuple[List[ImageBasisElement], torch.Tensor, Mapping[str, Any]]:
        spec_hash = _infer_spec_hash(spec)
        program_hash = program.hash(spec_hash)
        cache_key = (program_hash, str(device), str(dtype))
        if cache_key in self._compile_cache:
            elems_cached, group_ids, meta = self._compile_cache[cache_key]
            self._compile_cache.move_to_end(cache_key)
            elements = _restore_elements(elems_cached, device, dtype)
            return elements, group_ids.to(device=device), meta

        elements, group_ids, meta = compile_program_to_basis(program, spec, device)
        stored = _serialize_elements(elements) if (device.type == "cuda" and not cfg.cache_on_gpu) else elements
        group_ids_cpu = group_ids.detach().to("cpu") if torch.is_tensor(group_ids) else torch.tensor([], dtype=torch.long)
        self._compile_cache[cache_key] = (stored, group_ids_cpu, meta)
        if len(self._compile_cache) > cfg.compile_cache_size:
            self._compile_cache.popitem(last=False)
        return elements, group_ids, meta

    def _get_collocation(
        self,
        spec: object,
        device: torch.device,
        dtype: torch.dtype,
        cfg: RewardConfig,
        *,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        n_points, ratio_boundary = _resolve_collocation_params(cfg.n_points, cfg.ratio_boundary)
        spec_hash = _infer_spec_hash(spec)
        cache_key = (spec_hash, int(seed or 0), int(n_points), float(ratio_boundary), cfg.supervision_mode)
        if cache_key in self._colloc_cache:
            X_cpu, V_cpu, is_boundary = self._colloc_cache[cache_key]
            self._colloc_cache.move_to_end(cache_key)
            return (
                X_cpu.to(device=device, dtype=dtype),
                V_cpu.to(device=device, dtype=dtype),
                is_boundary.to(device=device) if is_boundary is not None else None,
            )

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
            pass_label="reward",
        )
        self._colloc_cache[cache_key] = (
            X_f.detach().to("cpu"),
            V_f.detach().to("cpu"),
            is_boundary_out.detach().to("cpu") if is_boundary_out is not None else None,
        )
        if len(self._colloc_cache) > cfg.collocation_cache_size:
            self._colloc_cache.popitem(last=False)
        return X_f, V_f, is_boundary_out

    def _empty_reward(
        self,
        program: Program,
        device: torch.device,
        dtype: torch.dtype,
        cfg: RewardConfig,
        latency_ms: float,
    ) -> RewardTerms:
        self.last_diagnostics["empty_compilation"] = True
        relerr = torch.tensor(1.0, device=device, dtype=dtype)
        instability = torch.tensor(cfg.nonfinite_penalty, device=device, dtype=dtype)
        complexity = _complexity_score(program, [], {"family_counts": {}}, device, dtype, cfg)
        novelty = torch.tensor(0.0, device=device, dtype=dtype)
        logR = torch.tensor(cfg.logR_clip[0], device=device, dtype=dtype)
        latency = torch.tensor(float(latency_ms), device=device, dtype=dtype)
        return RewardTerms(
            relerr=relerr,
            latency_ms=latency,
            instability=instability,
            complexity=complexity,
            novelty=novelty,
            logR=logR,
        )


def clip_log_reward(logR: torch.Tensor, clip_min: Optional[float], clip_max: Optional[float]) -> torch.Tensor:
    """Clip log-reward values for numerical stability."""
    if clip_min is None and clip_max is None:
        return logR
    min_val = clip_min if clip_min is not None else torch.finfo(logR.dtype).min
    max_val = clip_max if clip_max is not None else torch.finfo(logR.dtype).max
    return logR.clamp(min=min_val, max=max_val)


class _LatencyTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._start_time: Optional[float] = None
        self._elapsed: Optional[float] = None

    def start(self) -> None:
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self.device.type == "cuda" and self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            self._elapsed = float(self._start_event.elapsed_time(self._end_event))
        elif self._start_time is not None:
            self._elapsed = (time.perf_counter() - self._start_time) * 1000.0
        else:
            self._elapsed = 0.0

    def elapsed_ms(self) -> float:
        return float(self._elapsed or 0.0)

    def elapsed_ms_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.elapsed_ms(), device=device, dtype=dtype)


def _infer_spec_hash(spec: object) -> str:
    if isinstance(spec, str):
        return spec
    for attr in ("spec_hash", "hash", "id"):
        if hasattr(spec, attr):
            return str(getattr(spec, attr))
    return str(spec)


def _serialize_elements(elements: Sequence[ImageBasisElement]) -> List[Mapping[str, Any]]:
    return [elem.serialize() for elem in elements]


def _restore_elements(payload: Any, device: torch.device, dtype: torch.dtype) -> List[ImageBasisElement]:
    if not payload:
        return []
    if isinstance(payload[0], dict):
        return [ImageBasisElement.deserialize(data, device=device, dtype=dtype) for data in payload]
    return [elem for elem in payload]  # type: ignore[return-value]


def _matvec(A: Any, w: torch.Tensor, X: Optional[torch.Tensor]) -> torch.Tensor:
    if hasattr(A, "matvec"):
        return A.matvec(w, X)
    return A.matmul(w)


def _rel_resid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(target).clamp_min(1e-12)
    return torch.linalg.norm(pred - target) / denom


def _weight_range_penalty(weights: torch.Tensor) -> torch.Tensor:
    w_abs = weights.abs()
    if w_abs.numel() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    max_abs = w_abs.max()
    nonzero = w_abs[w_abs > 0]
    if nonzero.numel() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    min_abs = nonzero.min()
    ratio = max_abs / min_abs.clamp_min(1e-12)
    return torch.log10(ratio + 1.0)


def _condition_penalty(
    A: Any,
    X: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    cfg: RewardConfig,
) -> torch.Tensor:
    if not hasattr(A, "estimate_col_norms"):
        return torch.tensor(0.0, device=device, dtype=dtype)
    col_norms = getattr(A, "col_norms", None)
    if col_norms is None or not torch.is_tensor(col_norms):
        col_norms = A.estimate_col_norms(X)
    if col_norms.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    cond_ratio = col_norms.max() / col_norms.min().clamp_min(1e-6)
    threshold = torch.tensor(cfg.cond_threshold, device=device, dtype=dtype)
    penalty = torch.relu(torch.log10(cond_ratio.clamp_min(1.0)) - torch.log10(threshold))
    if torch.isfinite(penalty):
        return penalty
    return torch.tensor(cfg.nonfinite_penalty, device=device, dtype=dtype)


def _complexity_score(
    program: Program,
    elements: Sequence[ImageBasisElement],
    meta: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    cfg: RewardConfig,
) -> torch.Tensor:
    nodes = len(program.nodes)
    elem_count = len(elements)
    family_counts = meta.get("family_counts", {}) if isinstance(meta, dict) else {}
    family_count = len(family_counts) if isinstance(family_counts, dict) else 0
    w_nodes, w_elems, w_fams = cfg.complexity_weights
    score = w_nodes * nodes + w_elems * elem_count + w_fams * family_count
    return torch.tensor(float(score), device=device, dtype=dtype)


def _novelty_score(
    elements: Sequence[ImageBasisElement],
    weights: torch.Tensor,
    spec: object,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        system = ImageSystem(list(elements), weights.detach().to("cpu"))
        fp = structural_fingerprint(system, spec)
        novelty_val = float(novelty_lib.novelty_score(fp))
        return torch.tensor(novelty_val, device=device, dtype=dtype)
    except Exception:
        return torch.tensor(0.0, device=device, dtype=dtype)


def _combine_log_reward(
    relerr: torch.Tensor,
    latency_ms: torch.Tensor,
    instability: torch.Tensor,
    complexity: torch.Tensor,
    novelty: torch.Tensor,
    cfg: RewardConfig,
) -> torch.Tensor:
    relerr_safe = relerr.clamp_min(1e-12)
    log_relerr = torch.log(relerr_safe)
    w = cfg.weights
    logR = -w.alpha * log_relerr - w.beta * latency_ms - w.gamma * instability - w.delta * complexity + w.eta * novelty
    return logR


def _warmup_matvec(A: Any, X: torch.Tensor, device: torch.device, dtype: torch.dtype) -> None:
    try:
        if hasattr(A, "matvec"):
            w = torch.zeros((len(getattr(A, "elements", [])),), device=device, dtype=dtype)
            _ = A.matvec(w, X)
        else:
            w = torch.zeros((A.shape[1],), device=device, dtype=dtype)
            _ = A.matmul(w)
    except Exception:
        pass


__all__ = [
    "RewardTerms",
    "RewardNormalizer",
    "RewardWeights",
    "RewardConfig",
    "RewardComputer",
    "clip_log_reward",
]
