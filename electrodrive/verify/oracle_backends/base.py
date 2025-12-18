from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from ..oracle_types import OracleCost, OracleProvenance
from ..utils import get_git_sha, normalize_device, normalize_dtype, sha256_json, utc_now_iso


def _maybe_cuda_events(device: torch.device) -> Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
    except Exception:
        return None, None


def make_cost(
    start_wall: float,
    *,
    device: torch.device,
    start_event: Optional[torch.cuda.Event],
    end_event: Optional[torch.cuda.Event],
) -> OracleCost:
    wall_ms = max(0.0, (time.perf_counter() - start_wall) * 1000.0)
    cuda_ms = 0.0
    if start_event is not None and end_event is not None:
        try:
            torch.cuda.synchronize(device)
            cuda_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            cuda_ms = 0.0
    peak_vram_mb = 0.0
    try:
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    except Exception:
        peak_vram_mb = 0.0
    return OracleCost(wall_ms=wall_ms, cuda_ms=cuda_ms, peak_vram_mb=peak_vram_mb)


def make_provenance(device: torch.device, dtype: torch.dtype) -> OracleProvenance:
    device_norm = normalize_device(device)
    dtype_norm = normalize_dtype(dtype)
    device_name = "cpu"
    cuda_version = ""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = "cuda"
        cuda_version = torch.version.cuda or ""
    return OracleProvenance(
        git_sha=get_git_sha(),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        device_name=device_name,
        device=device_norm,
        dtype=dtype_norm,
        timestamp=utc_now_iso(),
    )


def fingerprint_config(name: str, cfg: Dict[str, Any]) -> str:
    return sha256_json({"name": name, "config": cfg})

