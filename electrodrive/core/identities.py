from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

FloatLike = Union[float, int]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _quantize_float(x: float, decimals: int) -> Union[float, str]:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    q = round(x, decimals)
    return 0.0 if q == 0.0 else q


def _to_primitive(obj: Any, *, float_decimals: int) -> Any:
    if torch is not None and isinstance(obj, getattr(torch, "Tensor", ())):
        arr = obj.detach().cpu().numpy()
        return _to_primitive(arr, float_decimals=float_decimals)

    if np is not None and isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _to_primitive(obj.item(), float_decimals=float_decimals)
        if np.issubdtype(obj.dtype, np.floating):
            v = obj.astype(np.float64)
            v = np.round(v, float_decimals)
            v[v == 0.0] = 0.0
            return v.tolist()
        if np.issubdtype(obj.dtype, np.integer):
            return obj.astype(np.int64).tolist()
        return obj.tolist()

    if np is not None and isinstance(obj, np.generic):
        py = obj.item()
        if isinstance(py, float):
            return _quantize_float(py, float_decimals)
        return py

    if dataclasses.is_dataclass(obj):
        return _to_primitive(dataclasses.asdict(obj), float_decimals=float_decimals)

    if isinstance(obj, Mapping):
        return {
            str(k): _to_primitive(v, float_decimals=float_decimals)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(obj, (list, tuple)):
        return [_to_primitive(x, float_decimals=float_decimals) for x in obj]

    if isinstance(obj, (set, frozenset)):
        return sorted(
            (_to_primitive(x, float_decimals=float_decimals) for x in obj),
            key=lambda x: json.dumps(x, sort_keys=True),
        )

    if _is_number(obj):
        return _quantize_float(float(obj), float_decimals)

    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()

    return str(obj)


def _canonical_json_bytes(obj: Any, *, float_decimals: int) -> bytes:
    prim = _to_primitive(obj, float_decimals=float_decimals)
    return json.dumps(
        prim,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(data: bytes, *, algo: str = "blake2b", size: int = 16) -> str:
    if algo == "blake2b":
        h = hashlib.blake2b(digest_size=size)
    elif algo == "sha256":
        h = hashlib.sha256()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported digest algo: {algo}")
    h.update(data)
    return h.hexdigest()


def compute_problem_hash(spec: Any, *, float_decimals: int = 12) -> str:
    try:
        if hasattr(spec, "to_json") and callable(getattr(spec, "to_json")):
            data = spec.to_json()
        elif isinstance(spec, str):
            try:
                data = json.loads(spec)
            except Exception:
                data = spec
        elif isinstance(spec, Mapping):
            data = spec
        elif hasattr(spec, "__dict__"):
            data = vars(spec)
        else:
            data = spec
    except Exception:
        data = str(spec)
    return _digest(_canonical_json_bytes(data, float_decimals=float_decimals))


def _to_np_array(a: Any, *, kind: str) -> "np.ndarray":
    if np is None:
        raise RuntimeError(f"numpy is required for mesh hashing ({kind})")
    if torch is not None and isinstance(a, getattr(torch, "Tensor", ())):
        return a.detach().cpu().numpy()
    if isinstance(a, np.ndarray):
        return a
    # Accept generic array-like inputs (lists, tuples, etc.) and coerce to ndarray.
    # This makes the hashing API robust for normal Python containers while still
    # giving a clear TypeError for genuinely unsupported inputs.
    try:
        return np.asarray(a)
    except Exception as e:  # pragma: no cover
        raise TypeError(
            f"{kind} must be an array-like or torch tensor"
        ) from e


def compute_mesh_hash(
    vertices: Any,
    faces: Any,
    *,
    float_decimals: int = 12,
) -> str:
    V = _to_np_array(vertices, kind="vertices").astype(float)
    F = _to_np_array(faces, kind="faces").astype(int)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (Nv,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must have shape (Nt,3)")

    if V.shape[0] == 0 or F.shape[0] == 0:
        return "mesh-empty"

    Vq = np.round(V, float_decimals)
    Vq[Vq == 0.0] = 0.0

    order = np.lexsort((Vq[:, 2], Vq[:, 1], Vq[:, 0]))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    V_sorted = Vq[order]

    Fm = inv[F]

    def _rotate_min(tri: "np.ndarray") -> "np.ndarray":
        i = int(np.argmin(tri))
        if i == 0:
            return tri
        if i == 1:
            return np.array([tri[1], tri[2], tri[0]], dtype=tri.dtype)
        return np.array([tri[2], tri[0], tri[1]], dtype=tri.dtype)

    F_norm = np.apply_along_axis(_rotate_min, 1, Fm)

    if F_norm.size > 0:
        tail = np.sort(F_norm[:, 1:3], axis=1)
        F_norm = np.column_stack([F_norm[:, 0], tail])

    max_idx = int(F_norm.max())
    if max_idx < (1 << 21):
        # Pack each triangle's sorted vertex indices into a 63-bit key:
        # 3 lanes × 21 bits. This gives a stable, lexicographic ordering
        # while remaining efficient for large meshes.
        key = (
            F_norm[:, 0].astype(np.int64) * (1 << 42)
            + F_norm[:, 1].astype(np.int64) * (1 << 21)
            + F_norm[:, 2].astype(np.int64)
        )
        idx = np.argsort(key, kind="mergesort")
    else:
        idx = np.lexsort((F_norm[:, 2], F_norm[:, 1], F_norm[:, 0]))

    F_sorted = F_norm[idx]

    payload = {"V": V_sorted.tolist(), "F": F_sorted.tolist()}
    return _digest(_canonical_json_bytes(payload, float_decimals=float_decimals))


def compute_kernel_hash(
    kernel_params: Mapping[str, Any],
    *,
    float_decimals: int = 12,
    whitelist: Optional[Iterable[str]] = None,
) -> str:
    if whitelist is not None:
        keys = sorted(str(k) for k in whitelist)
        data = {k: kernel_params[k] for k in keys if k in kernel_params}
    else:
        data = dict(sorted(kernel_params.items(), key=lambda kv: str(kv[0])))
    return _digest(_canonical_json_bytes(data, float_decimals=float_decimals))


@dataclass(frozen=True)
class RunIdentity:
    problem_hash: str
    mesh_hash: str
    kernel_hash: str
    hash_algo: str = dataclasses.field(default="blake2b-128")
    float_decimals: int = dataclasses.field(default=12)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[return-value]


def build_run_identity(
    *,
    spec: Any,
    vertices: Any,
    faces: Any,
    kernel_params: Mapping[str, Any],
    float_decimals: int = 12,
    whitelist: Optional[Iterable[str]] = None,
) -> RunIdentity:
    return RunIdentity(
        problem_hash=compute_problem_hash(spec, float_decimals=float_decimals),
        mesh_hash=compute_mesh_hash(vertices, faces, float_decimals=float_decimals),
        kernel_hash=compute_kernel_hash(
            kernel_params,
            float_decimals=float_decimals,
            whitelist=whitelist,
        ),
        hash_algo="blake2b-128",
        float_decimals=float_decimals,
    )


__all__ = [
    "compute_problem_hash",
    "compute_mesh_hash",
    "compute_kernel_hash",
    "RunIdentity",
    "build_run_identity",
]
