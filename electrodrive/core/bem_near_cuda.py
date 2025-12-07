# electrodrive/core/bem_near_cuda.py
from __future__ import annotations

import subprocess
import types
import warnings  # kept in case callers want to use it later
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.cpp_extension import CUDA_HOME, load as _load_ext

from electrodrive.utils.config import K_E as _K_E_DEFAULT

# Public API
__all__ = [
    "apply_near_quadrature_matvec_cuda",
    "is_bem_near_cuda_available",
    "get_bem_near_cuda_error",
]

# Cached loaded extension module
_EXT: Optional[types.ModuleType] = None
_EXT_COMPILE_ERROR: Optional[BaseException] = None


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _maybe_rewrite_compile_error(exc: BaseException) -> BaseException:
    """
    Make common build environment issues clearer (e.g., missing MSVC).
    """
    msg = str(exc)
    lower_msg = msg.lower()
    # Torch probes the MSVC compiler with `where cl`; surface that clearly.
    if "where', 'cl'" in lower_msg or ("cl.exe" in lower_msg and "not" in lower_msg):
        return RuntimeError(
            "MSVC (cl.exe) is not in PATH; run from a Visual Studio developer prompt "
            f"or ensure the Build Tools are installed. Original error: {msg}"
        )
    if "error checking compiler version for cl" in lower_msg:
        return RuntimeError(
            "MSVC (cl.exe) is not discoverable; activate a Visual Studio developer "
            f"environment. Original error: {msg}"
        )
    return exc


def _probe_nvcc() -> Tuple[bool, Optional[str]]:
    """
    Check whether nvcc is available on PATH when CUDA_HOME is unset.
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        detail = (result.stdout or result.stderr or "").strip()
        return True, detail or None
    except FileNotFoundError:
        return False, "nvcc not found on PATH"
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        detail = (exc.stderr or exc.stdout or "").strip() or str(exc)
        return False, detail


def _load_extension(device: torch.device) -> types.ModuleType:
    """
    JIT-compile and load the bem_near CUDA extension.

    This compiles bem_near_cuda.cpp + bem_near_cuda_kernel.cu into a
    single binary module (bem_near_cuda_ext) and caches it globally.
    """
    global _EXT, _EXT_COMPILE_ERROR

    if _EXT is not None:
        return _EXT
    if _EXT_COMPILE_ERROR is not None:
        raise _EXT_COMPILE_ERROR

    src_dir = _this_dir()
    sources = [
        str(src_dir / "bem_near_cuda.cpp"),
        str(src_dir / "bem_near_cuda_kernel.cu"),
    ]

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math"]

    # Try to specialise for the active GPU (e.g., your 5090 / Blackwell).
    # Guard against very old or very new/bogus compute capabilities that nvcc
    # does not support (for example, reported as compute_12 or compute_120).
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(device)
            arch_int = int(f"{major}{minor}")
            # Only add an explicit gencode for architectures in a conservative range.
            # For current toolchains, 50–90 covers modern supported SM versions.
            if 50 <= arch_int <= 90:
                extra_cuda_cflags.append(
                    f"-gencode=arch=compute_{arch_int},code=sm_{arch_int}"
                )
            # If arch_int is outside this range (e.g., 120 on newer GPUs),
            # do not add an explicit gencode; fall back to the defaults chosen
            # by torch.utils.cpp_extension / nvcc.
        except Exception:
            # Best-effort; fall back to PyTorch defaults
            pass

    try:
        _EXT = _load_ext(
            name="bem_near_cuda_ext",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        return _EXT
    except BaseException as exc:  # noqa: BLE001
        friendly = _maybe_rewrite_compile_error(exc)
        _EXT_COMPILE_ERROR = friendly
        raise friendly


def is_bem_near_cuda_available(device: Optional[torch.device] = None) -> bool:
    """
    Best-effort probe: can we compile and load the CUDA near-field extension?

    This will attempt a one-time JIT build; failures are memoized so we
    do not spam compilations on every call.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return False

    global _EXT, _EXT_COMPILE_ERROR

    # Quick preflight: allow builds when CUDA_HOME is unset but nvcc exists.
    if CUDA_HOME is None:
        has_nvcc, detail = _probe_nvcc()
        if not has_nvcc:
            if _EXT_COMPILE_ERROR is None:
                msg = "CUDA toolkit not found (set CUDA_HOME or ensure nvcc is on PATH)"
                if detail:
                    msg = f"{msg}; nvcc probe: {detail}"
                _EXT_COMPILE_ERROR = RuntimeError(msg)
            return False

    if _EXT is not None:
        return True

    try:
        _load_extension(device)
        return True
    except BaseException:
        return False


def get_bem_near_cuda_error() -> Optional[str]:
    """
    Return the cached compilation/load error message for the CUDA extension.
    """
    if _EXT_COMPILE_ERROR is None:
        return None
    return str(_EXT_COMPILE_ERROR)


def apply_near_quadrature_matvec_cuda(
    V_far: Tensor,
    sigma: Tensor,
    *,
    centroids: Tensor,
    areas: Tensor,
    panel_vertices: np.ndarray | Tensor,
    near_pairs: np.ndarray | Tensor,
    quad_order: int,
    K_E: float | None = None,
) -> Tensor:
    """
    CUDA near-field quadrature correction for the BEM matvec.

    This is the GPU analogue of `_apply_near_quadrature_matvec` in
    `electrodrive.core.bem_quadrature`, but executes the entire
    (i, j) near-pair loop on the GPU using a custom CUDA kernel.

    Parameters
    ----------
    V_far:
        [N] CUDA tensor; far-field result from bem_matvec_gpu.
    sigma:
        [N] CUDA tensor; panel charge density.
    centroids:
        [N, 3] CUDA tensor of panel centroids.
    areas:
        [N] CUDA tensor of panel areas.
    panel_vertices:
        Either a NumPy array of shape [N, 3, 3] from TriMesh, or a
        CUDA tensor with the same shape and dtype as centroids.
    near_pairs:
        Either a NumPy array of shape [P, 2] (int64) with (i, j) near
        interaction indices, or a CUDA int64 tensor with the same.
    quad_order:
        Integer quadrature order (e.g. 2) understood by your CUDA kernel.
    K_E:
        Coulomb constant to use. If None, uses the global K_E from
        electrodrive.utils.config.

    Returns
    -------
    V_out:
        Corrected potential with the same shape and dtype as V_far.
    """
    # Nothing to do if there are no near pairs
    if near_pairs is None:
        return V_far
    if isinstance(near_pairs, np.ndarray) and near_pairs.size == 0:
        return V_far
    if isinstance(near_pairs, Tensor) and near_pairs.numel() == 0:
        return V_far

    if V_far.device.type != "cuda":
        raise ValueError(
            "apply_near_quadrature_matvec_cuda expects V_far on CUDA; "
            f"got device={V_far.device}"
        )

    device = V_far.device
    dtype = V_far.dtype

    # Choose Coulomb constant
    if K_E is None:
        K_E = float(_K_E_DEFAULT)
    else:
        K_E = float(K_E)

    # Normalise inputs: all geometry on CUDA, contiguous
    if isinstance(panel_vertices, np.ndarray):
        panel_vertices_t = torch.as_tensor(
            panel_vertices, device=device, dtype=dtype
        )
    else:
        panel_vertices_t = panel_vertices.to(
            device=device, dtype=dtype, non_blocking=True
        )
    panel_vertices_t = panel_vertices_t.contiguous()

    if isinstance(near_pairs, np.ndarray):
        near_pairs_t = torch.as_tensor(
            near_pairs, device=device, dtype=torch.int64
        )
    else:
        near_pairs_t = near_pairs.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
    near_pairs_t = near_pairs_t.contiguous()

    centroids_t = centroids.to(
        device=device, dtype=dtype, non_blocking=True
    ).contiguous()
    areas_t = areas.to(
        device=device, dtype=dtype, non_blocking=True
    ).contiguous()
    sigma_t = sigma.to(
        device=device, dtype=dtype, non_blocking=True
    ).contiguous()
    V_far_t = V_far.to(
        device=device, dtype=dtype, non_blocking=True
    ).contiguous()

    # Use the CUDA extension; let callers (bem.py) handle any fallback.
    ext = _load_extension(device)
    V_out = ext.near_quadrature_matvec(
        V_far_t,
        sigma_t,
        centroids_t,
        areas_t,
        panel_vertices_t,
        near_pairs_t,
        float(K_E),
        int(quad_order),
    )
    # Preserve original shape in case V_far is e.g. [N] vs [N,]
    return V_out.reshape_as(V_far_t)
