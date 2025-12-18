from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Any

import torch


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_sha(cwd: str | None = None) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"
    return sha or "unknown"


def normalize_device(device: str | torch.device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, str):
        return str(torch.device(device))
    raise TypeError(f"Unsupported device type: {type(device)!r}")


def normalize_dtype(dtype: str | torch.dtype) -> str:
    if isinstance(dtype, torch.dtype):
        name = str(dtype)
    elif isinstance(dtype, str):
        name = dtype
    else:
        raise TypeError(f"Unsupported dtype type: {type(dtype)!r}")
    return name.replace("torch.", "")


def dtype_from_str(name: str) -> torch.dtype:
    key = name.replace("torch.", "").lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[key]


def device_matches(device: str | torch.device, tensor_device: torch.device) -> bool:
    dev = torch.device(device)
    if dev.type != tensor_device.type:
        return False
    if dev.index is None:
        return True
    return dev.index == tensor_device.index


def require_cuda(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (GPU-first rule)")


def hash_tensor(tensor: torch.Tensor) -> str:
    require_cuda(tensor, "tensor")
    t = tensor.detach().contiguous()
    t_cpu = t.cpu()
    h = hashlib.sha256()
    h.update(str(t_cpu.dtype).encode("utf-8"))
    h.update(str(tuple(t_cpu.shape)).encode("utf-8"))
    h.update(t_cpu.numpy().tobytes())
    return h.hexdigest()
