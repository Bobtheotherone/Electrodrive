from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import sys

import torch

from . import VERIFY_VERSION
from .utils import get_git_sha, hash_tensor, require_cuda, sha256_bytes, sha256_json


@dataclass(frozen=True)
class CacheKey:
    version: str
    spec_key: str
    oracle_key: str
    points_key: Optional[str] = None

    def to_string(self) -> str:
        parts = [self.version, self.spec_key, self.oracle_key]
        if self.points_key:
            parts.append(self.points_key)
        return "|".join(parts)

    def filename(self) -> str:
        return sha256_bytes(self.to_string().encode("utf-8")) + ".json"


@dataclass
class VerifyCacheConfig:
    ram_max_items: int = 128
    ram_max_bytes: int = 256 * 1024 * 1024
    max_tensor_bytes: int = 16 * 1024 * 1024
    enable_ram: bool = True
    enable_disk: bool = True
    disk_root: Path = field(default_factory=lambda: Path("artifacts/verify_cache"))
    disk_write_tensors: bool = False
    allow_large_tensors: bool = False
    version_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.version_fingerprint:
            sha = get_git_sha()
            self.version_fingerprint = sha if sha != "unknown" else VERIFY_VERSION


@dataclass
class CacheEntry:
    value: Any
    meta: Dict[str, object]
    size_bytes: int


class VerifyCache:
    def __init__(self, config: Optional[VerifyCacheConfig] = None) -> None:
        self.config = config or VerifyCacheConfig()
        self._ram: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._ram_bytes = 0
        if self.config.enable_disk:
            self.config.disk_root.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        spec: Dict[str, object],
        oracle_name: str,
        oracle_fingerprint: str,
        *,
        points: Optional[torch.Tensor] = None,
        include_points: bool = False,
    ) -> CacheKey:
        spec_key = sha256_json(spec)
        oracle_key = sha256_json({"name": oracle_name, "fingerprint": oracle_fingerprint})
        points_key = hash_tensor(points) if include_points and points is not None else None
        return CacheKey(
            version=self.config.version_fingerprint,
            spec_key=spec_key,
            oracle_key=oracle_key,
            points_key=points_key,
        )

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        key_str = key.to_string()
        if self.config.enable_ram and key_str in self._ram:
            entry = self._ram.pop(key_str)
            self._ram[key_str] = entry
            return entry
        if self.config.enable_disk:
            return self._load_disk(key)
        return None

    def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        meta: Optional[Dict[str, object]] = None,
        allow_large: bool = False,
        write_disk: bool = False,
    ) -> bool:
        self._ensure_cuda_value(value, name="cache_value")
        entry_meta = dict(meta or {})
        entry_meta.setdefault("cache_key", key.to_string())
        entry_meta.setdefault("version", key.version)
        entry_meta.setdefault("provenance", {})
        size_bytes = self._estimate_size_bytes(value)
        if torch.is_tensor(value):
            if not self._allow_tensor_store(size_bytes, allow_large=allow_large):
                return False
        if self.config.enable_ram:
            self._insert_ram(key.to_string(), CacheEntry(value=value, meta=entry_meta, size_bytes=size_bytes))
        if write_disk and self.config.enable_disk:
            self._write_disk(key, CacheEntry(value=value, meta=entry_meta, size_bytes=size_bytes))
        return True

    def _allow_tensor_store(self, size_bytes: int, *, allow_large: bool) -> bool:
        if self.config.allow_large_tensors or allow_large:
            return True
        return size_bytes <= self.config.max_tensor_bytes

    def _estimate_size_bytes(self, value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, (list, tuple)):
            return int(sum(self._estimate_size_bytes(v) for v in value))
        if isinstance(value, dict):
            return int(sum(self._estimate_size_bytes(v) for v in value.values()))
        return int(sys.getsizeof(value))

    def _ensure_cuda_value(self, value: Any, *, name: str) -> None:
        if torch.is_tensor(value):
            require_cuda(value, name)
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if torch.is_tensor(item):
                    require_cuda(item, f"{name}[{idx}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if torch.is_tensor(item):
                    require_cuda(item, f"{name}.{key}")

    def _insert_ram(self, key: str, entry: CacheEntry) -> None:
        if key in self._ram:
            old = self._ram.pop(key)
            self._ram_bytes -= old.size_bytes
        self._ram[key] = entry
        self._ram_bytes += entry.size_bytes
        self._evict_ram_if_needed()

    def _evict_ram_if_needed(self) -> None:
        while self._ram and (
            len(self._ram) > self.config.ram_max_items or self._ram_bytes > self.config.ram_max_bytes
        ):
            _, entry = self._ram.popitem(last=False)
            self._ram_bytes -= entry.size_bytes

    def _write_disk(self, key: CacheKey, entry: CacheEntry) -> None:
        payload: Dict[str, object] = {
            "key": key.to_string(),
            "meta": entry.meta,
        }
        if torch.is_tensor(entry.value):
            if not self.config.disk_write_tensors:
                return
            tensor_path = self.config.disk_root / key.filename().replace(".json", ".pt")
            torch.save(entry.value, tensor_path)
            payload["tensor_path"] = str(tensor_path)
        else:
            if self._is_jsonable(entry.value):
                payload["value"] = entry.value
            else:
                return
        out_path = self.config.disk_root / key.filename()
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_disk(self, key: CacheKey) -> Optional[CacheEntry]:
        path = self.config.disk_root / key.filename()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        meta = dict(payload.get("meta", {}))
        if "tensor_path" in payload:
            if not self.config.disk_write_tensors:
                return None
            tensor_path = Path(payload["tensor_path"])
            if not tensor_path.exists():
                return None
            value = torch.load(tensor_path, map_location="cuda")
            if torch.is_tensor(value):
                require_cuda(value, "cache_value")
        else:
            value = payload.get("value", None)
        entry = CacheEntry(value=value, meta=meta, size_bytes=self._estimate_size_bytes(value))
        if self.config.enable_ram:
            self._insert_ram(key.to_string(), entry)
        return entry

    @staticmethod
    def _is_jsonable(value: Any) -> bool:
        try:
            json.dumps(value)
            return True
        except Exception:
            return False
