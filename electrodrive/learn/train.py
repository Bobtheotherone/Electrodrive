# electrodrive/learn/train.py
"""
Training orchestration for electrodrive PINN / MoI style models.

This module is intentionally self-contained and robust so that test
suites can import it directly without requiring the full CLI stack.

Key features:
- Supports:
  - Mixed precision (AMP) via torch.amp / torch.cuda.amp (auto-detected).
  - Optional torch.compile with graceful fallback when unavailable.
  - Microbatching and gradient accumulation (accum_steps).
  - Autotuning of points_per_step based on available VRAM.
  - "Loose" curriculum configs that are mapped into the expected schema.
- Emits JSONL metrics compatible with simple log processors.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim

# -----------------------------------------------------------------------------
# GradScaler import (prefer modern torch.amp API with safe fallback)
# -----------------------------------------------------------------------------
try:
    # PyTorch >= 2.0 style
    from torch.amp import GradScaler  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older PyTorch
    # Backward-compatible fallback
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]

# -----------------------------------------------------------------------------
# AMP autocast import with graceful fallback
# -----------------------------------------------------------------------------
try:
    # Preferred modern API (PyTorch >= 2.0)
    from torch.amp import autocast  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older PyTorch
    from torch.cuda.amp import autocast  # type: ignore[no-redef]


from electrodrive.learn.specs import (
    CurriculumSpec,
    DatasetSpec,
    EvalSpec,
    ExperimentConfig,
    ModelSpec,
    TrainerSpec,
)
from electrodrive.learn.dataset import build_dataloaders
from electrodrive.learn.encoding import ENCODING_DIM
from electrodrive.learn.models.pinn_harmonic import PINNHarmonic
from electrodrive.learn.models.moi_symbolic import MoISymbolic

logger = logging.getLogger("EDE.Learn.Train")


# =============================================================================
# Utility logging
# =============================================================================


def _log_info(event: str, **kv: Any) -> None:
    """
    Structured info logger compatible with stdlib logging.

    We serialize payload as JSON in the message instead of passing arbitrary
    kwargs to logger.info, which would raise TypeError in stdlib logging.
    """
    safe: Dict[str, Any] = {"event": event}
    for k, v in kv.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                safe[k] = float(v)
            except Exception:
                safe[k] = v
        else:
            safe[k] = v

    try:
        logger.info("%s", json.dumps(safe))
    except Exception:
        # Fallback: best-effort stringification
        logger.info("%s %s", event, safe)


def _log_entry(
    step: int,
    metrics: Dict[str, float],
    lr: float | None = None,
    start_time: float | None = None,
    file_path: Path | None = None,
    prefix: str = "",
) -> None:
    """
    Append a single metrics record to a JSONL file (if provided).

    - step: global step number
    - metrics: dictionary of metric_name -> value
    - lr: learning rate (optional)
    - start_time: if provided, elapsed_time is computed
    - prefix: optional metric name prefix (e.g. "val_")
    """
    data: Dict[str, Any] = {"step": int(step)}
    if lr is not None:
        data["lr"] = float(lr)
    if start_time is not None:
        data["elapsed_time"] = float(time.time() - start_time)

    for k, v in metrics.items():
        data[prefix + k] = float(v)

    if file_path is not None:
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:  # pragma: no cover
            logger.exception("Failed to write metrics to %s", file_path)


def _save_ckpt(
    model: torch.nn.Module,
    optim_: optim.Optimizer,
    step: int,
    path: Path,
) -> None:
    """
    Save a minimal training checkpoint.

    This function is intentionally forgiving: it never raises to callers.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": int(step),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim_.state_dict(),
            },
            path,
        )
    except Exception:  # pragma: no cover - IO dependent
        logger.exception("Failed to save checkpoint to %s", path)


# =============================================================================
# Model factory
# =============================================================================


def initialize_model(config: ExperimentConfig) -> torch.nn.Module:
    """
    Instantiate model based on ExperimentConfig.

    Supported types:
      - "pinn_harmonic": default PINN harmonic model.
      - "pinn_harmonic_large": large residual/checkpointed variant.
      - "moi_symbolic": symbolic Method-of-Images style model.
    """
    mtype = config.model.model_type
    params = config.model.params

    if mtype == "pinn_harmonic":
        return PINNHarmonic(params)
    if mtype == "pinn_harmonic_large":
        # Large variant behind explicit name; keep defaults untouched for
        # other model types.
        large_params = {
            "width": 1536,
            "depth": 16,
            "residual_every": 4,
            "gradient_checkpointing": True,
        }
        if params:
            large_params.update(params)
        return PINNHarmonic(large_params)
    if mtype == "moi_symbolic":
        return MoISymbolic(params)

    raise ValueError(f"Unknown model type: {mtype}")


# =============================================================================
# AMP / compile / autotune helpers
# =============================================================================


def _select_autocast_dtype(device: torch.device) -> Tuple[bool, torch.dtype | None, str]:
    """
    Choose an autocast dtype given the device.

    Returns:
        (enabled, dtype, mode_str)
        where mode_str in {"bf16", "fp16", "off"}.
    """
    if device.type != "cuda":
        return False, None, "off"

    amp_dtype: torch.dtype | None = None
    mode = "off"

    is_bf16_supported = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        with contextlib.suppress(Exception):
            is_bf16_supported = bool(torch.cuda.is_bf16_supported())

    if is_bf16_supported and hasattr(torch, "bfloat16"):
        amp_dtype = torch.bfloat16
        mode = "bf16"
    else:
        if hasattr(torch, "float16"):
            amp_dtype = torch.float16
            mode = "fp16"

    if amp_dtype is None:
        return False, None, "off"

    return True, amp_dtype, mode


def _maybe_compile(
    model: torch.nn.Module,
    compile_flag: bool | str,
    mode: str,
) -> torch.nn.Module:
    """
    Optionally wrap model with torch.compile.

    Behavior:
    - If compile_flag is falsy: return model unchanged.
    - If compile_flag is a string, treat it as the requested mode.
    - If compile_flag is True, use provided 'mode' or "reduce-overhead".
    - If torch.compile is unavailable or fails at runtime, print a note
      and fall back to the original model.

    This is intentionally soft so tests run on environments without torch.compile.
    """
    if not compile_flag:
        return model

    if isinstance(compile_flag, str):
        requested_mode = compile_flag
    else:
        requested_mode = mode or "reduce-overhead"

    if not hasattr(torch, "compile"):
        print("EDE.Learn: torch.compile requested but not available; running uncompiled.")
        return model

    try:
        compiled = torch.compile(model, mode=requested_mode, fullgraph=False)
        print(f"EDE.Learn: torch.compile enabled (mode={requested_mode}).")
        return compiled
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"EDE.Learn: torch.compile failed ({exc}); running uncompiled.")
        return model


_AUTOTUNE_CANDIDATES = [1024, 2048, 4096, 8192, 16384, 32768]
_AUTOTUNE_POINTS_CACHE: Dict[Tuple[Any, ...], int] = {}


def _autotune_points_per_step(
    model: torch.nn.Module,
    device: torch.device,
    *,
    floor: int = 16384,
    ceil: int = 262144,
    target_util: float = 0.8,
) -> int:
    """
    Probe-based microbatch autotune.

    Runs lightweight forward+backward passes over synthetic batches for
    multiple candidate sizes, measures peak GPU memory, and picks the largest
    candidate that fits under target_util of the reported free memory. Results
    are cached per (device, VRAM, model signature, amp mode, compile flag).
    """
    min_candidate = min(_AUTOTUNE_CANDIDATES)
    candidate_floor = max(1, min(int(floor), min_candidate))
    candidate_ceil = max(int(ceil), min_candidate)
    candidates = [
        c for c in _AUTOTUNE_CANDIDATES if candidate_floor <= c <= candidate_ceil
    ] or [min_candidate]
    cpu_default = max(int(floor), min_candidate)
    safe_default = min_candidate

    # AMP + dtype selection mirrors the main training loop as closely as possible.
    use_amp, amp_dtype, amp_mode = _select_autocast_dtype(device)
    use_scaler = bool(use_amp and amp_dtype == torch.float16)

    compiled_flag = bool(
        getattr(model, "_compiled", False)
        or getattr(model, "_is_compiled", False)
        or "Compiled" in type(model).__name__
        or "torch._dynamo" in repr(type(model))
    )
    model_type = type(model).__name__
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        n_params = 0

    # Device metadata (best-effort; tolerate missing CUDA)
    device_name = str(device)
    total_bytes = None
    free_bytes = None
    total_gb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            device_name = torch.cuda.get_device_name(device)
        with contextlib.suppress(Exception):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if total_bytes is None:
            with contextlib.suppress(Exception):
                props = torch.cuda.get_device_properties(device)
                total_bytes = getattr(props, "total_memory", None)
        if total_bytes:
            total_gb = float(total_bytes) / (1024.0 ** 3)

    cache_key = (device_name, total_gb, model_type, n_params, amp_mode, compiled_flag)
    if cache_key in _AUTOTUNE_POINTS_CACHE:
        return _AUTOTUNE_POINTS_CACHE[cache_key]

    # CPU or unknown CUDA memory info: return conservative default immediately.
    if device.type != "cuda" or not torch.cuda.is_available() or free_bytes is None:
        _AUTOTUNE_POINTS_CACHE[cache_key] = cpu_default
        return cpu_default

    max_allowed_bytes = int(free_bytes * float(target_util))
    if max_allowed_bytes <= 0:
        _AUTOTUNE_POINTS_CACHE[cache_key] = safe_default
        return safe_default

    try:
        sample_param = next(model.parameters())
        param_dtype = sample_param.dtype
    except Exception:
        param_dtype = torch.float32

    loss_weights = dict(TrainerSpec().loss_weights)
    scaler = GradScaler(enabled=use_scaler)
    model_mode = model.training
    best_valid: int | None = None

    def _make_synth_batch(points: int) -> Dict[str, torch.Tensor]:
        encoding = torch.zeros(
            (points, ENCODING_DIM), device=device, dtype=param_dtype
        )
        return {
            "X": torch.zeros((points, 3), device=device, dtype=param_dtype),
            "V_gt": torch.zeros(points, device=device, dtype=param_dtype),
            "is_boundary": torch.zeros(points, device=device, dtype=torch.bool),
            "mask_finite": torch.ones(points, device=device, dtype=torch.bool),
            "encoding": encoding,
        }

    try:
        model.train()
        for points in candidates:
            peak_bytes = None
            batch: Dict[str, torch.Tensor] | None = None
            losses: Dict[str, torch.Tensor] | None = None
            loss: torch.Tensor | None = None

            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats(device)

            try:
                batch = _make_synth_batch(points)
                with autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    losses = model.compute_loss(batch, loss_weights)
                    if isinstance(losses, dict):
                        loss = losses.get("total", None)
                    else:
                        loss = losses
                    if loss is None or not torch.is_tensor(loss):
                        raise RuntimeError("compute_loss must return a tensor under 'total'")
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(device)
                with contextlib.suppress(Exception):
                    peak_bytes = torch.cuda.max_memory_allocated(device)
            except RuntimeError:
                # Treat all runtime errors (including OOM) as invalid candidates.
                peak_bytes = None
            except Exception:
                peak_bytes = None
            finally:
                with contextlib.suppress(Exception):
                    model.zero_grad(set_to_none=True)
                del batch, losses, loss
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()

            if peak_bytes is None:
                continue
            if peak_bytes <= max_allowed_bytes:
                best_valid = points
    finally:
        model.train(model_mode)

    chosen = best_valid if best_valid is not None else safe_default
    _AUTOTUNE_POINTS_CACHE[cache_key] = chosen
    return chosen


# =============================================================================
# Curriculum handling (loose schema -> canonical)
# =============================================================================

_GEOMETRY_KEYS = {"plane", "sphere", "cylinder2D", "wedge", "mesh"}


def _to_attr_namespace(obj: Any) -> Any:
    """
    Recursively convert dicts to SimpleNamespace for attribute-style access.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_attr_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_attr_namespace(v) for v in obj)
    return obj


def _to_plain_dict(obj: Any) -> Any:
    """
    Recursively convert SimpleNamespace to plain dict; passthrough for others.
    """
    if isinstance(obj, SimpleNamespace):
        return {k: _to_plain_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(v) for v in obj)
    return obj


def _normalize_curriculum_dict(d: dict) -> dict:
    """
    Turn a 'loose' curriculum dict with top-level geometry weights into:

        {
            "stages": [
                {
                    "name": "auto_stage",
                    "mixture": [
                        {"geometry": "plane", "weight": 1.0, "params": {...}},
                        ...
                    ]
                }
            ],
            ...other keys...
        }
    """
    geo = {k: v for k, v in d.items() if k in _GEOMETRY_KEYS}
    if not geo:
        return d

    stage = {"name": "auto_stage", "mixture": []}
    for g, v in geo.items():
        if isinstance(v, (int, float)):
            stage["mixture"].append({"geometry": g, "weight": float(v), "params": {}})
        elif isinstance(v, dict):
            w = float(v.get("weight", 1.0))
            params = {kk: vv for kk, vv in v.items() if kk != "weight"}
            stage["mixture"].append({"geometry": g, "weight": w, "params": params})
        else:
            stage["mixture"].append({"geometry": g, "weight": 1.0, "params": {}})

    rest = {k: v for k, v in d.items() if k not in _GEOMETRY_KEYS}
    return {"stages": [stage], **rest}


def _ensure_curriculum_defaults(ns: SimpleNamespace) -> SimpleNamespace:
    """
    Ensure the curriculum namespace has all attributes expected downstream.
    """
    if not hasattr(ns, "include_pdf_sources"):
        ns.include_pdf_sources = False

    # Stages / mixture normalization
    if not hasattr(ns, "stages") or ns.stages is None:
        ns.stages = [SimpleNamespace(name="auto_stage", mixture=[])]
    else:
        fixed_stages = []
        for st in ns.stages:
            st_ns = st if isinstance(st, SimpleNamespace) else _to_attr_namespace(st)
            if not hasattr(st_ns, "name"):
                st_ns.name = "stage"
            if not hasattr(st_ns, "mixture") or st_ns.mixture is None:
                st_ns.mixture = []
            else:
                fixed_mix = []
                for m in st_ns.mixture:
                    m_ns = m if isinstance(m, SimpleNamespace) else _to_attr_namespace(m)
                    if not hasattr(m_ns, "geometry"):
                        m_ns.geometry = "plane"
                    if not hasattr(m_ns, "weight"):
                        m_ns.weight = 1.0
                    if not hasattr(m_ns, "params") or m_ns.params is None:
                        m_ns.params = {}
                    else:
                        if isinstance(m_ns.params, SimpleNamespace):
                            m_ns.params = _to_plain_dict(m_ns.params)
                        elif not isinstance(m_ns.params, dict):
                            try:
                                m_ns.params = dict(m_ns.params)  # type: ignore[arg-type]
                            except Exception:
                                m_ns.params = {}
                    fixed_mix.append(m_ns)
                st_ns.mixture = fixed_mix
            fixed_stages.append(st_ns)
        ns.stages = fixed_stages

    # geometry_weights as plain dict
    default_geo_weights = {
        "plane": 1.0,
        "sphere": 1.0,
        "cylinder2D": 1.0,
        "wedge": 1.0,
        "mesh": 0.0,
    }
    if not hasattr(ns, "geometry_weights") or ns.geometry_weights is None:
        ns.geometry_weights = dict(default_geo_weights)
    else:
        if isinstance(ns.geometry_weights, SimpleNamespace):
            ns.geometry_weights = _to_plain_dict(ns.geometry_weights)
        elif not isinstance(ns.geometry_weights, dict):
            try:
                ns.geometry_weights = dict(ns.geometry_weights)  # type: ignore[arg-type]
            except Exception:
                ns.geometry_weights = dict(default_geo_weights)

    # parallel_planes as plain dict
    default_pp = {
        "enabled": False,
        "separation": 2.0,
        "max_images": 16,
    }
    if not hasattr(ns, "parallel_planes") or ns.parallel_planes is None:
        ns.parallel_planes = dict(default_pp)
    else:
        if isinstance(ns.parallel_planes, SimpleNamespace):
            ns.parallel_planes = _to_plain_dict(ns.parallel_planes)
        elif not isinstance(ns.parallel_planes, dict):
            try:
                ns.parallel_planes = dict(ns.parallel_planes)  # type: ignore[arg-type]
            except Exception:
                ns.parallel_planes = dict(default_pp)

    return ns


def _finalize_curriculum_for_dataset(cur: Any) -> Any:
    """
    Normalize curriculum object into something dataset code can consume.
    """
    if isinstance(cur, CurriculumSpec):
        return cur

    if isinstance(cur, dict):
        cur = _to_attr_namespace(cur)

    if isinstance(cur, SimpleNamespace):
        cur = _ensure_curriculum_defaults(cur)

        # Ensure dicts where needed
        if hasattr(cur, "geometry_weights") and isinstance(cur.geometry_weights, SimpleNamespace):
            cur.geometry_weights = _to_plain_dict(cur.geometry_weights)
        if hasattr(cur, "parallel_planes") and isinstance(cur.parallel_planes, SimpleNamespace):
            cur.parallel_planes = _to_plain_dict(cur.parallel_planes)

        if hasattr(cur, "geometry_weights") and not isinstance(cur.geometry_weights, dict):
            cur.geometry_weights = _to_plain_dict(cur.geometry_weights)
        if hasattr(cur, "parallel_planes") and not isinstance(cur.parallel_planes, dict):
            cur.parallel_planes = _to_plain_dict(cur.parallel_planes)

        return cur

    # Unknown type; return as-is.
    return cur


def _ingest_curriculum(curriculum: Any) -> Any:
    """
    Accepts either:
    - CurriculumSpec
    - schema-compliant dictionary
    - "loose" dictionary with top-level geometry keys
    - SimpleNamespace with approximately expected fields

    Returns an object consumable by dataset builders (either CurriculumSpec
    or a SimpleNamespace emulating its attributes).
    """
    if not isinstance(curriculum, dict):
        return _finalize_curriculum_for_dataset(curriculum)

    norm = _normalize_curriculum_dict(curriculum)

    # Try strict CurriculumSpec first
    try:
        spec = CurriculumSpec(**norm)
        return _finalize_curriculum_for_dataset(spec)
    except Exception:
        # Fallback: attribute-style loose spec
        ns = _to_attr_namespace(norm)
        ns = _ensure_curriculum_defaults(ns)
        logger.warning(
            "Using loose curriculum schema (attribute adapter). Keys=%s",
            list(curriculum.keys()),
        )
        return _finalize_curriculum_for_dataset(ns)


def _subsample_collocation_batch(
    batch: Dict[str, Any],
    points_per_step: int,
    boundary_fraction: float,
) -> Dict[str, Any]:
    """
    Boundary-aware subsampling for collocation batches.

    If points_per_step is smaller than the batch, sample indices without
    replacement while attempting to include a fraction of boundary points
    when is_boundary is provided. Applies the chosen indices to every
    per-point tensor (leading dim == N).
    """
    X = batch.get("X")
    if X is None or not torch.is_tensor(X):
        return batch

    if not hasattr(X, "shape") or X.ndim == 0:
        return batch

    N = int(X.shape[0])
    if N == 0 or points_per_step <= 0 or points_per_step >= N:
        return batch

    device = X.device

    def _safe_empty_like(idx_like: torch.Tensor) -> torch.Tensor:
        return idx_like.new_empty((0,), dtype=idx_like.dtype, device=idx_like.device)

    is_boundary = batch.get("is_boundary")
    if (
        torch.is_tensor(is_boundary)
        and hasattr(is_boundary, "shape")
        and is_boundary.ndim >= 1
        and is_boundary.shape[0] == N
    ):
        boundary_fraction = max(0.0, min(1.0, float(boundary_fraction)))
        boundary_idx = torch.nonzero(is_boundary, as_tuple=True)[0]
        interior_idx = torch.nonzero(~is_boundary, as_tuple=True)[0]

        n_bnd_target = int(points_per_step * boundary_fraction)
        n_bnd = min(n_bnd_target, boundary_idx.numel())
        n_int = min(points_per_step - n_bnd, interior_idx.numel())

        if n_bnd > 0:
            perm_bnd = torch.randperm(boundary_idx.numel(), device=boundary_idx.device)[:n_bnd]
            chosen_bnd = boundary_idx[perm_bnd]
        else:
            chosen_bnd = _safe_empty_like(boundary_idx)

        if n_int > 0:
            perm_int = torch.randperm(interior_idx.numel(), device=interior_idx.device)[:n_int]
            chosen_int = interior_idx[perm_int]
        else:
            chosen_int = _safe_empty_like(interior_idx)

        idx = torch.cat([chosen_bnd, chosen_int], dim=0)

        remaining = points_per_step - idx.numel()
        if remaining > 0:
            all_idx = torch.arange(N, device=device)
            if idx.numel() > 0:
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[idx] = False
                reservoir = all_idx[mask]
            else:
                reservoir = all_idx

            if reservoir.numel() >= remaining:
                perm_extra = torch.randperm(reservoir.numel(), device=device)[:remaining]
                extra = reservoir[perm_extra]
            else:
                # If we still need more, allow reuse as a last resort.
                extra = torch.randint(0, N, (remaining,), device=device)

            idx = torch.cat([idx, extra], dim=0)

        if idx.numel() > 1:
            perm = torch.randperm(idx.numel(), device=device)
            idx = idx[perm]
    else:
        idx = torch.randperm(N, device=device)[:points_per_step]

    new_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if (
            torch.is_tensor(v)
            and hasattr(v, "shape")
            and v.ndim >= 1
            and v.shape[0] == N
        ):
            new_batch[k] = v[idx]
        else:
            new_batch[k] = v
    return new_batch


def _to_device(
    batch: Dict[str, Any],
    device: torch.device,
    non_blocking: bool,
) -> Dict[str, Any]:
    """Move tensors in batch to target device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(
                device=device,
                non_blocking=non_blocking,
            )
        else:
            out[k] = v
    return out


# =============================================================================
# Validation
# =============================================================================


def validate(
    model: torch.nn.Module,
    dl,
    trainer: TrainerSpec,
    use_amp: bool,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """
    Run validation loop and return averaged losses.
    """
    model.eval()
    device = next(model.parameters()).device

    _, amp_dtype, _ = _select_autocast_dtype(device)
    non_blocking = bool(
        device.type == "cuda"
        and getattr(
            trainer,
            "pin_memory",
            False,
        )
    )

    agg: Dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            if not batch:
                continue

            batch = _to_device(
                batch,
                device,
                non_blocking=non_blocking,
            )
            for key in ("X", "V_gt"):
                if key in batch:
                    batch[key] = batch[key].to(dtype=dtype)

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                losses = model.compute_loss(batch, trainer.loss_weights)

            for k, v in losses.items():
                if torch.is_tensor(v):
                    agg[k] = agg.get(k, 0.0) + float(v.detach())
                else:
                    agg[k] = agg.get(k, 0.0) + float(v)

            n_batches += 1

    if n_batches > 0:
        for k in list(agg.keys()):
            agg[k] /= float(n_batches)

    return agg


# =============================================================================
# Main train() entry point
# =============================================================================


def train(
    config: ExperimentConfig,
    output_dir: Path,
) -> int:
    """
    High-level training entry point used by tests and CLI.

    Returns:
        0 on success, non-zero on hard failure.
    """
    # Normalize config sub-sections that may be raw dicts.
    if isinstance(config.dataset, dict):
        config.dataset = DatasetSpec(**config.dataset)
    if isinstance(config.model, dict):
        config.model = ModelSpec(**config.model)
    if isinstance(config.trainer, dict):
        config.trainer = TrainerSpec(**config.trainer)
    if isinstance(config.evaluation, dict):
        config.evaluation = EvalSpec(**config.evaluation)

    # Curriculum: accept loose dictionaries as well.
    config.curriculum = _ingest_curriculum(config.curriculum)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    trainer: TrainerSpec = config.trainer

    device = torch.device(config.device)
    train_dtype = getattr(torch, config.train_dtype)
    non_blocking = bool(
        device.type == "cuda"
        and getattr(
            trainer,
            "pin_memory",
            False,
        )
    )

    # Trainer knobs (with backward-compatible defaults)
    amp_cfg = getattr(trainer, "amp", None)
    compile_flag = getattr(trainer, "compile", False)
    compile_mode = getattr(trainer, "compile_mode", "reduce-overhead")
    accum_steps = int(getattr(trainer, "accum_steps", 1)) or 1
    user_points_per_step = int(getattr(trainer, "points_per_step", 0))
    boundary_fraction = float(getattr(trainer, "boundary_fraction", 0.1))

    # Seed RNGs
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    # Resolve AMP mode
    if amp_cfg in ("bf16", "fp16"):
        if device.type == "cuda":
            if amp_cfg == "bf16" and hasattr(torch, "bfloat16"):
                use_amp = True
                amp_dtype = torch.bfloat16
                amp_mode = "bf16"
            else:
                use_amp = True
                amp_dtype = torch.float16
                amp_mode = "fp16"
        else:
            use_amp, amp_dtype, amp_mode = False, None, "off"
    else:
        use_amp, amp_dtype, amp_mode = _select_autocast_dtype(device)

    # Build dataloaders
    logger.info("Building dataloaders...")
    try:
        train_loader, val_loader = build_dataloaders(config)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Dataloader init failed: %s", e, exc_info=True)
        return 1

    if len(train_loader) == 0:
        logger.error("Training loader is empty.")
        return 1

    batches_per_epoch = max(len(train_loader), 1)
    steps_per_epoch = math.ceil(batches_per_epoch / max(accum_steps, 1))
    total_steps = trainer.max_epochs * steps_per_epoch
    warmup_steps = int(getattr(trainer, "warmup_frac", 0.0) * total_steps)

    # Initialize model
    model = initialize_model(config).to(device=device, dtype=train_dtype)
    n_params = sum(p.numel() for p in model.parameters())

    # Optional compile
    model = _maybe_compile(model, compile_flag, compile_mode)

    # Autotune points_per_step
    if user_points_per_step > 0:
        points_per_step = user_points_per_step
        autotuned = False
    else:
        points_per_step = _autotune_points_per_step(model, device)
        autotuned = True

    # VRAM telemetry (best-effort)
    cuda_free_gb = cuda_total_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            free_b, total_b = torch.cuda.mem_get_info(device)
            cuda_free_gb = free_b / (1024**3)
            cuda_total_gb = total_b / (1024**3)

    compiled_flag = bool(
        getattr(model, "_compiled", False)
        or "Compiled" in type(model).__name__
        or "torch._dynamo" in repr(type(model))
    )

    _log_info(
        "model_init",
        model_type=config.model.model_type,
        params=n_params,
        amp=amp_mode,
        compile_active=compiled_flag,
        points_per_step=points_per_step,
        points_per_step_autotuned=bool(autotuned),
        accum_steps=accum_steps,
        effective_batch_size=points_per_step * accum_steps,
        cuda_free_gb=(f"{cuda_free_gb:.2f}" if cuda_free_gb is not None else None),
        cuda_total_gb=(f"{cuda_total_gb:.2f}" if cuda_total_gb is not None else None),
    )

    # Optimizer and scheduler
    optim_ = optim.AdamW(
        model.parameters(),
        lr=trainer.learning_rate,
        weight_decay=trainer.weight_decay,
    )

    if trainer.lr_scheduler == "cosine" and total_steps > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=total_steps)
    else:
        scheduler = None

    _log_info(
        "scheduler_init",
        batches_per_epoch=batches_per_epoch,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # AMP GradScaler (only for fp16)
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    step = 0
    start_time = time.time()
    best_val = float("inf")
    patience = 0

    _log_info(
        "train_start",
        total_steps=total_steps,
        amp=amp_mode,
        use_grad_scaler=use_scaler,
        accum_steps=accum_steps,
    )

    # =========================
    # Training loop
    # =========================
    for epoch in range(trainer.max_epochs):
        model.train()
        accum_counter = 0

        for batch_idx, batch in enumerate(train_loader):
            if not batch:
                continue

            batch = _subsample_collocation_batch(
                batch,
                points_per_step=points_per_step,
                boundary_fraction=boundary_fraction,
            )

            # Move to device
            batch = _to_device(
                batch,
                device,
                non_blocking=non_blocking,
            )

            # Cast selected tensors to train_dtype
            for key in ("X", "V_gt"):
                if key in batch:
                    batch[key] = batch[key].to(dtype=train_dtype)

            if accum_counter == 0:
                optim_.zero_grad(set_to_none=True)

            try:
                with autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    losses = model.compute_loss(batch, trainer.loss_weights)
                    loss = losses["total"]
                    if accum_steps > 1:
                        loss = loss / float(accum_steps)
            except Exception as e:  # pragma: no cover - model-specific
                logger.error("Error at training step %d: %s", step, e, exc_info=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1

            if accum_counter >= accum_steps:
                # Optional gradient clipping
                if trainer.grad_clip_norm > 0:
                    if use_scaler:
                        scaler.unscale_(optim_)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        trainer.grad_clip_norm,
                    )

                # Optimizer step
                if use_scaler:
                    scaler.step(optim_)
                    scaler.update()
                else:
                    optim_.step()

                if scheduler is not None:
                    scheduler.step()

                accum_counter = 0
                step += 1

                if step % trainer.log_every_n_steps == 0:
                    metrics: Dict[str, float] = {}
                    for k, v in losses.items():
                        if torch.is_tensor(v):
                            metrics[k] = float(v.detach())
                        else:
                            metrics[k] = float(v)
                    _log_entry(
                        step,
                        metrics,
                        lr=optim_.param_groups[0]["lr"],
                        start_time=start_time,
                        file_path=metrics_file,
                    )

        # Flush leftover grads at epoch end
        if accum_counter > 0:
            if trainer.grad_clip_norm > 0:
                if use_scaler:
                    scaler.unscale_(optim_)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    trainer.grad_clip_norm,
                )

            if use_scaler:
                scaler.step(optim_)
                scaler.update()
            else:
                optim_.step()

            if scheduler is not None:
                scheduler.step()

            step += 1
            accum_counter = 0

        # Validation
        if len(val_loader) > 0 and (epoch + 1) % trainer.val_every_n_epochs == 0:
            val_metrics = validate(
                model,
                val_loader,
                trainer,
                use_amp=use_amp,
                dtype=train_dtype,
            )

            _log_entry(
                step,
                val_metrics,
                prefix="val_",
                file_path=metrics_file,
            )

            cur_val = float(val_metrics.get("total", float("inf")))
            if cur_val < best_val:
                best_val = cur_val
                patience = 0
                _save_ckpt(
                    model,
                    optim_,
                    step,
                    output_dir / "ckpt_best.pt",
                )
                _log_info(
                    "val_improved",
                    epoch=epoch + 1,
                    best_val=best_val,
                )
            else:
                patience += 1
                if (
                    trainer.early_stopping_patience > 0
                    and patience >= trainer.early_stopping_patience
                ):
                    _log_info(
                        "early_stopping",
                        epoch=epoch + 1,
                        best_val=best_val,
                    )
                    break

        # Periodic checkpoint at epoch boundary
        if (epoch + 1) % trainer.ckpt_every_n_epochs == 0:
            _save_ckpt(
                model,
                optim_,
                step,
                output_dir / f"ckpt_epoch_{epoch + 1}.pt",
            )

    logger.info("Training finished.")
    _save_ckpt(
        model,
        optim_,
        step,
        output_dir / "ckpt_last.pt",
    )

    if best_val < float("inf"):
        _log_entry(
            step,
            {"total": best_val},
            prefix="final_",
            file_path=metrics_file,
        )

    return 0
