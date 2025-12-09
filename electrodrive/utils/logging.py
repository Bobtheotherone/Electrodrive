from __future__ import annotations

import dataclasses
import datetime as _dt
import io
import json
import logging
import math
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# --------------------------------------------
# JSON utilities (NaN/Inf safe + compact)
# --------------------------------------------


def _json_sanitize(v: Any) -> Any:
    """
    Convert values into JSON-safe primitives.

    Rules:
    - Finite floats are emitted as-is.
    - NaN / ±Inf floats are stringified ("NaN", "Infinity", "-Infinity")
      so they never break json.dumps.
    - torch.Tensors:
        * small (<= 1024 elements): full .tolist()
        * large: summarized with shape/dtype/min/max/mean
    - Containers are handled recursively.
    - Anything else that json.dumps can't handle is stringified.
    """
    # floats
    try:
        if isinstance(v, float):
            if math.isfinite(v):
                return v
            if math.isnan(v):
                return "NaN"
            return "Infinity" if v > 0 else "-Infinity"
    except Exception:
        pass

    # tensors
    if torch is not None:
        try:
            if isinstance(v, torch.Tensor):
                t = v.detach()
                # Small tensors: full materialization
                if t.numel() <= 1024:
                    return t.cpu().tolist()
                # Large tensors: summary only, to avoid gigantic logs
                try:
                    t_cpu = t.detach().cpu()
                    # nan-safe stats if available
                    if hasattr(torch, "nanmin"):
                        t_min = float(torch.nanmin(t_cpu).item())
                        t_max = float(torch.nanmax(t_cpu).item())
                        t_mean = float(torch.nanmean(t_cpu.float()).item())
                    else:
                        t_min = float(t_cpu.min().item())
                        t_max = float(t_cpu.max().item())
                        t_mean = float(t_cpu.float().mean().item())
                    return {
                        "_type": "tensor_summary",
                        "shape": list(t_cpu.shape),
                        "dtype": str(t.dtype),
                        "min": t_min,
                        "max": t_max,
                        "mean": t_mean,
                    }
                except Exception:
                    # Last resort: just shape/dtype
                    try:
                        return {
                            "_type": "tensor_summary",
                            "shape": list(t.shape),
                            "dtype": str(t.dtype),
                        }
                    except Exception:
                        return "Tensor"
        except Exception:
            pass

    # basic containers
    if isinstance(v, dict):
        return {str(k): _json_sanitize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_sanitize(x) for x in v]
    if isinstance(v, (set, frozenset)):
        # sets are unordered; sort their sanitized representation for stability
        return sorted(_json_sanitize(x) for x in v)

    # other objects
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def _json_dump_line(obj: Dict[str, Any]) -> str:
    """
    Dump a single JSON object to a compact UTF-8 JSON string, after sanitization.
    """
    return json.dumps(_json_sanitize(obj), separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------
# Perf flags (stable API)
# --------------------------------------------


@dataclass(frozen=True)
class RuntimePerfFlags:
    amp: bool = False
    train_dtype: str = "float32"  # for learn stack
    compile: bool = False         # request torch.compile() where available
    tf32: str = "off"             # "off" | "high" | "medium" | "highest"


# --------------------------------------------
# JSONL Logger (append-only, thread-safe)
# --------------------------------------------


class JsonlLogger:
    """
    Minimal, robust JSONL event logger.

    - Safe for NaN/Inf; values are sanitized.
    - Safe for torch tensors; large tensors are summarized.
    - Never raises to callers (best-effort, failures are swallowed).
    - .info/.debug/.warning/.error all write a single JSON object per line.
    - Adds "ts", "level", "msg" fields plus any structured k/v pairs.
    - Adds SmartLog helpers (health/efficiency/accuracy/logic).
    """

    def __init__(self, out_dir: Path | str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self._stream: Optional[io.TextIOBase] = None
        self._open()

    # ----- context manager support -----
    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- file handling -----
    def _open(self) -> None:
        """
        Best-effort open of the events.jsonl file in append mode.
        """
        try:
            self._stream = self.path.open("a", encoding="utf-8")
        except Exception:
            self._stream = None

    def close(self) -> None:
        """
        Close the underlying stream; future writes will attempt to reopen.
        """
        try:
            if self._stream:
                try:
                    self._stream.flush()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
        finally:
            # Mark as closed so _emit can attempt to reopen on the next write.
            self._stream = None

    # ------------- Core write -------------
    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        rec: Dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": level,
            "msg": msg,
        }
        if fields:
            rec.update(fields)

        # Try to build a JSON line, but never raise to caller.
        try:
            line = _json_dump_line(rec)
        except Exception:
            # last-resort: stringify everything
            try:
                rec2 = {str(k): str(v) for k, v in rec.items()}
                line = json.dumps(rec2, ensure_ascii=False)
            except Exception:
                # truly hopeless; give up silently
                return

        with self._lock:
            try:
                if self._stream is None:
                    self._open()
                if self._stream:
                    self._stream.write(line + "\n")
                    self._stream.flush()
            except Exception:
                # swallow any IO errors; logging must never break the caller
                return

    # ------------- Public API (level helpers) -------------
    def info(self, msg: str, **fields: Any) -> None:
        self._emit("INFO", msg, **fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("DEBUG", msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("WARN", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        """
        Log an error. If the caller passes exc_info=True, attach traceback text
        into a "trace" field but do not re-raise.
        """
        exc_info_flag = fields.get("exc_info", False)
        if exc_info_flag:
            try:
                import traceback

                fields["trace"] = traceback.format_exc()
            except Exception:
                pass
            # Avoid json-serializing the raw exc_info value itself
            fields.pop("exc_info", None)
        self._emit("ERROR", msg, **fields)

    # --------------------------------------------
    # SmartLog helpers (health/efficiency/accuracy/logic)
    # --------------------------------------------
    def smart_progress(self, **fields: Any) -> None:
        """
        Generic progress probe (loss/residuals/iter/epoch/throughput).
        Use keys like: step, epoch, loss, resid_true_l2, resid_bc, samples_per_sec, etc.
        """
        fields.setdefault("type", "smart_progress")
        self._emit("INFO", "Smart progress", **fields)

    def smart_health(self, **fields: Any) -> None:
        """
        System health snapshot (gpu_mem_alloc_mb, gpu_mem_reserved_mb, tf32_mode, dtype, etc).
        Automatically augments with basic VRAM stats if CUDA is available.
        """
        if torch is not None and hasattr(torch, "cuda"):
            try:
                if torch.cuda.is_available():
                    dev = torch.cuda.current_device()
                    fields.setdefault(
                        "gpu_mem_alloc_mb",
                        float(torch.cuda.memory_allocated(dev)) / (1024.0 * 1024.0),
                    )
                    fields.setdefault(
                        "gpu_mem_reserved_mb",
                        float(torch.cuda.memory_reserved(dev)) / (1024.0 * 1024.0),
                    )
            except Exception:
                pass
        fields.setdefault("type", "smart_health")
        self._emit("INFO", "Smart health", **fields)

    def smart_efficiency(self, **fields: Any) -> None:
        """
        Efficiency probe (tiles_per_sec, batches_per_sec, autotile, occupancy proxies, etc.).
        """
        fields.setdefault("type", "smart_efficiency")
        self._emit("DEBUG", "Smart efficiency", **fields)

    def smart_accuracy(self, **fields: Any) -> None:
        """
        Accuracy snapshot (bc_linf, dual_l2, pde_linf, energy_rel_diff, etc.).
        """
        fields.setdefault("type", "smart_accuracy")
        self._emit("INFO", "Smart accuracy", **fields)

    def smart_logic(self, **fields: Any) -> None:
        """
        Logical checks (max_principle_margin, reciprocity_dev, invariances, etc.).
        """
        fields.setdefault("type", "smart_logic")
        self._emit("INFO", "Smart logic", **fields)

    def phase_start(self, name: str, **fields: Any) -> None:
        """
        Mark the start of a logical phase/section of the run.
        """
        self._emit("INFO", "Phase start", phase=name, **fields)

    def phase_end(self, name: str, **fields: Any) -> None:
        """
        Mark the end of a logical phase/section of the run.
        """
        self._emit("INFO", "Phase end", phase=name, **fields)


# --------------------------------------------
# TF32 helpers + runtime environment logging
# --------------------------------------------


def get_effective_tf32_mode() -> str:
    """
    Best-effort query of the effective TF32 mode for matmul.

    Returns:
        "off" / "high" / "medium" / "highest" / "unavailable" / "unknown"
    """
    if torch is None:
        return "unavailable"
    # New-style API (PyTorch 1.12+)
    try:
        if hasattr(torch, "get_float32_matmul_precision"):
            mode = str(torch.get_float32_matmul_precision())
            if mode:
                return mode
    except Exception:
        pass
    # Legacy CUDA matmul knobs
    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
            matmul = getattr(torch.backends.cuda, "matmul", None)
            if matmul is not None:
                if getattr(matmul, "allow_tf32", False):
                    return "high"
                return "off"
    except Exception:
        pass
    return "unknown"


def log_runtime_environment(
    logger: JsonlLogger,
    perf_flags: Optional[RuntimePerfFlags] = None,
) -> None:
    """
    Log a best-effort summary of the runtime environment.

    Includes:
      - Python version / executable
      - Platform
      - numpy version (if available)
      - torch version / CUDA availability / device summary
      - TF32 effective mode
      - default torch dtype
      - perf_flags (if provided)
      - EDE_RUN_DIR (if set)
    """
    v: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    # Optional: environment variables relevant to this stack
    ede_run_dir = os.environ.get("EDE_RUN_DIR")
    if ede_run_dir:
        v["ede_run_dir"] = ede_run_dir

    # numpy
    try:
        import numpy as _np  # type: ignore

        v["numpy"] = _np.__version__
    except Exception:
        v["numpy"] = "unavailable"

    # torch / CUDA
    if torch is not None:
        v["torch"] = getattr(torch, "__version__", "unknown")
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        v["cuda_available"] = cuda_ok
        if cuda_ok:
            try:
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                v["device_name"] = props.name
                v["total_memory_gb"] = float(props.total_memory) / (1024.0**3)
            except Exception:
                pass
        v["tf32_effective"] = get_effective_tf32_mode()
        try:
            v["default_dtype"] = str(torch.get_default_dtype())
        except Exception:
            pass

    # perf flags (e.g., amp / train_dtype / compile / tf32)
    if perf_flags is not None:
        try:
            v["perf_flags"] = dataclasses.asdict(perf_flags)
        except Exception:
            # last resort: string form
            v["perf_flags"] = str(perf_flags)

    logger.info("Runtime environment.", **v)


def log_peak_vram(logger: JsonlLogger, phase: str = "run") -> None:
    """
    Log peak VRAM usage so far.

    Fields:
      - phase: logical name for the stage (e.g. "bem_probe", "train", "eval")
      - peak_vram_mb: torch.cuda.max_memory_allocated
      - peak_vram_reserved_mb: torch.cuda.max_memory_reserved
    """
    rec: Dict[str, Any] = {"phase": phase}
    if torch is not None and hasattr(torch, "cuda"):
        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                rec["peak_vram_mb"] = float(torch.cuda.max_memory_allocated(dev)) / (
                    1024.0 * 1024.0
                )
                rec["peak_vram_reserved_mb"] = float(
                    torch.cuda.max_memory_reserved(dev)
                ) / (1024.0 * 1024.0)
        except Exception:
            # Leave rec with whatever we have
            pass
    logger.info("VRAM peak usage.", **rec)
