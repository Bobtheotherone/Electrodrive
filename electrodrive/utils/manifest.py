from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid_now() -> str:
    ts_ms = int(time.time() * 1000)
    rnd = uuid.uuid4().int & ((1 << 80) - 1)
    x = (ts_ms << 80) | rnd
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[x & 0x1F])
        x >>= 5
    return "".join(reversed(out))


@dataclass
class DeviceInfo:
    cuda_available: bool
    device_name: str
    total_memory_gb: float
    tf32_effective_mode: str
    tf32: bool
    dtype: str
    device_index: int
    device_count: int
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class Manifest:
    schema_version: str
    run_id: str
    created_at_utc: str
    user: str
    code_commit: Optional[str]
    identities: Dict[str, Any]
    kernel: Dict[str, Any]
    device: DeviceInfo
    env: Dict[str, Any]


def _detect_device() -> DeviceInfo:
    cuda = bool(
        torch is not None
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
    )
    name = "cpu"
    mem_gb = 0.0
    tf32_mode = "unavailable"
    dtype = "float32"
    idx = 0
    ndev = 0
    tver: Optional[str] = None
    cver: Optional[str] = None

    if torch is not None:
        try:
            dtype = str(torch.get_default_dtype())
        except Exception:
            dtype = "unknown"
        try:
            if hasattr(torch, "get_float32_matmul_precision"):
                tf32_mode = str(torch.get_float32_matmul_precision())
        except Exception:
            tf32_mode = "unavailable"
        try:
            tver = getattr(torch, "__version__", None)
        except Exception:
            tver = None
        try:
            if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
                cver = torch.version.cuda  # type: ignore[attr-defined]
        except Exception:
            cver = None

    if cuda and torch is not None:
        try:
            idx = int(torch.cuda.current_device())
            ndev = int(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(idx)
            name = props.name
            mem_gb = float(props.total_memory) / (1024.0 ** 3)
        except Exception:
            name = "unknown"

    tf32_bool = tf32_mode.lower() not in (
        "off",
        "highest",
        "unavailable",
        "unknown",
    )

    return DeviceInfo(
        cuda_available=cuda,
        device_name=name,
        total_memory_gb=mem_gb,
        tf32_effective_mode=tf32_mode,
        tf32=tf32_bool,
        dtype=dtype,
        device_index=idx,
        device_count=ndev,
        torch_version=tver,
        cuda_version=cver,
    )


def _build_manifest_internal(
    *,
    identities: Dict[str, Any],
    kernel: Dict[str, Any],
    code_commit: Optional[str],
    extra_env: Optional[Dict[str, Any]],
    run_id: Optional[str],
) -> Manifest:
    run_id = run_id or _ulid_now()
    device = _detect_device()
    env: Dict[str, Any] = dict(extra_env or {})
    try:
        v = os.sys.version_info
        env.setdefault("python_version", f"{v.major}.{v.minor}.{v.micro}")
    except Exception:
        env.setdefault("python_version", "unknown")
    env.setdefault("platform", os.name)
    return Manifest(
        schema_version="1.0",
        run_id=run_id,
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        user=os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        code_commit=code_commit,
        identities=identities,
        kernel=kernel,
        device=device,
        env=env,
    )


def build_manifest(
    *,
    identities: Dict[str, Any],
    kernel: Dict[str, Any],
    code_commit: Optional[str] = None,
    extra_env: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Manifest:
    return _build_manifest_internal(
        identities=identities,
        kernel=kernel,
        code_commit=code_commit,
        extra_env=extra_env,
        run_id=run_id,
    )


def write_manifest(path: Path, manifest: Manifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = {
        "schema_version": manifest.schema_version,
        "run_id": manifest.run_id,
        "created_at_utc": manifest.created_at_utc,
        "user": manifest.user,
        "code_commit": manifest.code_commit,
        "identities": manifest.identities,
        "kernel": manifest.kernel,
        "device": asdict(manifest.device),
        "env": manifest.env,
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    try:
        os.replace(tmp, path)
    except Exception:
        pass
    return path
