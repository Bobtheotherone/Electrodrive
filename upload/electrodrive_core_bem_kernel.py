from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

try:
    import torch  # type: ignore
    from torch import Tensor
except Exception as e:  # pragma: no cover
    raise ImportError("Torch is required by bem_kernel.py") from e

from electrodrive.utils.logging import JsonlLogger
from electrodrive.utils.config import K_E

# Optional KeOps
USE_KEOPS = False
try:  # pragma: no cover
    from pykeops.torch import LazyTensor  # type: ignore

    USE_KEOPS = True
except Exception:  # pragma: no cover
    LazyTensor = None  # type: ignore


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _ensure_tile_size(tile_size: int) -> int:
    """Clamp tile_size to a sensible positive integer."""
    if tile_size is None or tile_size <= 0:
        return 1024
    return int(tile_size)


def _log_tensor_stats(
    logger: Optional[JsonlLogger],
    name: str,
    x: Tensor,
    level: str = "debug",
) -> None:
    """
    Log basic finite-range stats for a tensor (min, max, any_nonfinite).

    This is intentionally lightweight and only used when a logger is provided.
    """
    if logger is None:
        return
    try:
        finite = torch.isfinite(x)
        any_nonfinite = not bool(torch.all(finite))
        x_finite = x[finite]
        if x_finite.numel() == 0:
            stats = dict(any_nonfinite=any_nonfinite, numel=int(x.numel()))
        else:
            stats = dict(
                any_nonfinite=any_nonfinite,
                numel=int(x.numel()),
                min=float(x_finite.min().item()),
                max=float(x_finite.max().item()),
            )
        if level == "debug":
            logger.debug(f"{name} stats.", **stats)  # type: ignore[arg-type]
        elif level == "info":
            logger.info(f"{name} stats.", **stats)  # type: ignore[arg-type]
        elif level == "warning":
            logger.warning(f"{name} stats.", **stats)  # type: ignore[arg-type]
    except Exception:
        # Stats are best-effort only; never let them break the solver.
        return


# --------------------------------------------------------------------------------------
# Pluggable physics: single-layer Green's function abstraction
# --------------------------------------------------------------------------------------


@runtime_checkable
class SingleLayerKernel(Protocol):
    """
    Interface for a single-layer Green's function kernel.

    Implementations must provide:
      - potential(diff, r): scalar kernel K_ij for potential
      - e_field_weight(diff, r): scalar kernel W_ij such that

            E_i = sum_j W_ij * (sigma_j * A_j) * (x_i - r_j)

    Shapes:
        diff : (T, N, 3)
        r    : (T, N)
        returns (T, N)
    """

    name: str

    def potential(self, diff: Tensor, r: Tensor) -> Tensor:
        ...

    def e_field_weight(self, diff: Tensor, r: Tensor) -> Tensor:
        ...


@dataclass
class LaplaceSingleLayerKernel:
    """
    Free-space Laplace single-layer kernel:

        G(r) = K_E / |r|
    """

    name: str = "laplace_single_layer"

    def potential(self, diff: Tensor, r: Tensor) -> Tensor:
        # diff is unused for Laplace but kept for interface consistency
        return K_E / r

    def e_field_weight(self, diff: Tensor, r: Tensor) -> Tensor:
        # Weight such that:
        #   E(x) = sum_j W_ij * (sigma_j * A_j) * (x_i - r_j)
        inv_r3 = 1.0 / (r * r * r)
        return K_E * inv_r3


DEFAULT_SINGLE_LAYER_KERNEL: SingleLayerKernel = LaplaceSingleLayerKernel()


def have_keops() -> bool:
    """Return True if the KeOps backend is importable."""
    return bool(USE_KEOPS)


# --------------------------------------------------------------------------------------
# Differentiable cores (pure-Torch): used by diff paths and by non-diff wrappers.
# --------------------------------------------------------------------------------------


def _bem_matvec_core_torch(
    centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    self_integrals: Optional[Tensor] = None,
    tile_size: int = 1024,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable matrix-vector apply for the single-layer operator:

        (V_ind)_i = sum_j K_ij * (sigma_j * A_j)

    where K_ij is provided by `kernel.potential(...)`.

    Diagonal handling:
    - If self_integrals is provided, K_ii is replaced by self_integrals[i].
    - If self_integrals is None, K_ii is set to 0.0 (consistent with FMM/KeOps).
    """
    device, dtype = sigma.device, sigma.dtype
    N = int(sigma.shape[0])

    tile_size = _ensure_tile_size(tile_size)
    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL

    V = torch.zeros(N, device=device, dtype=dtype)
    if N == 0:
        return V

    # Basic sanity: shapes
    assert centroids.shape[0] == N, "centroids and sigma must have the same length"
    assert areas.shape[0] == N, "areas and sigma must have the same length"

    for i0 in range(0, N, tile_size):
        i1 = min(i0 + tile_size, N)
        chunk = centroids[i0:i1]  # (T,3)
        diff = chunk[:, None, :] - centroids[None, :, :]  # (T,N,3)
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)  # (T,N)

        # Pluggable physics prior
        K = ker.potential(diff, r)  # (T,N)

        # Diagonal indices within this chunk
        arng = torch.arange(i0, i1, device=device)
        local_idx = torch.arange(i1 - i0, device=device)

        if self_integrals is not None:
            # Replace diagonal with analytic integrals
            K[local_idx, arng] = self_integrals[arng]
        else:
            # Explicitly zero the diagonal to avoid huge 1/eps terms.
            # This ensures consistency with FMM/KeOps backends.
            K[local_idx, arng] = 0.0

        V[i0:i1] = torch.sum(K * sigma[None, :] * areas[None, :], dim=1)
    return V


def _bem_potential_targets_core_torch(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable induced potential at arbitrary targets due to panel charges.

        V(target_i) = sum_j K_ij * (sigma_j * A_j)
    """
    device, dtype = targets.device, targets.dtype
    M, N = int(targets.shape[0]), int(sigma.shape[0])

    tile_size = _ensure_tile_size(tile_size)
    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL

    V = torch.zeros(M, device=device, dtype=dtype)
    if M == 0 or N == 0:
        return V

    assert src_centroids.shape[0] == N, "src_centroids and sigma must have the same length"
    assert areas.shape[0] == N, "areas and sigma must have the same length"

    for i0 in range(0, M, tile_size):
        i1 = min(i0 + tile_size, M)
        chunk = targets[i0:i1]  # (T,3)
        diff = chunk[:, None, :] - src_centroids[None, :, :]  # (T,N,3)
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)  # (T,N)
        K = ker.potential(diff, r)
        V[i0:i1] = torch.sum(K * sigma[None, :] * areas[None, :], dim=1)
    return V


def _bem_E_field_targets_core_torch(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable induced E-field at arbitrary targets due to panel charges.

        E(x) = sum_j W_ij * (sigma_j * A_j) * (x_i - r_j)

    where W_ij = kernel.e_field_weight(...).

    Returns
    -------
    Tensor of shape [M, 3]
    """
    device, dtype = targets.device, targets.dtype
    M, N = int(targets.shape[0]), int(sigma.shape[0])

    tile_size = _ensure_tile_size(tile_size)
    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL

    E = torch.zeros(M, 3, device=device, dtype=dtype)
    if M == 0 or N == 0:
        return E

    assert src_centroids.shape[0] == N, "src_centroids and sigma must have the same length"
    assert areas.shape[0] == N, "areas and sigma must have the same length"

    for i0 in range(0, M, tile_size):
        i1 = min(i0 + tile_size, M)
        chunk = targets[i0:i1]  # (T,3)
        diff = chunk[:, None, :] - src_centroids[None, :, :]  # (T,N,3)
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)  # (T,N)

        weight = ker.e_field_weight(diff, r)  # (T,N)
        w = weight * sigma[None, :] * areas[None, :]  # (T,N)
        # Weighted sum over sources
        E[i0:i1] = torch.sum(w[..., None] * diff, dim=1)
    return E


# --------------------------------------------------------------------------------------
# Non-diff implementations (best-effort KeOps, else tiled Torch); safe fallbacks
# --------------------------------------------------------------------------------------


def _bem_matvec_torch_tiled(
    sigma: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    tile_size: int,
    self_integrals: Optional[Tensor],
    logger: Optional[JsonlLogger] = None,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Pure Torch non-differentiable matvec with explicit tiling.

    Kept as a separate helper so we can force this path in tests if needed.
    """
    device = sigma.device
    tile_size = _ensure_tile_size(tile_size)
    N = int(sigma.shape[0])

    with torch.no_grad():
        _log_tensor_stats(logger, "bem_matvec_torch_tiled_sigma", sigma, level="debug")
        _log_tensor_stats(logger, "bem_matvec_torch_tiled_areas", areas, level="debug")

        V = _bem_matvec_core_torch(
            centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            self_integrals=self_integrals,
            tile_size=tile_size,
            kernel=kernel,
        )

        # Safety check: clamp any non-finite outputs to zero and log.
        if not torch.all(torch.isfinite(V)):
            if logger is not None:
                logger.error(
                    "Non-finite entries detected in _bem_matvec_torch_tiled output; "
                    "clamping to zero.",
                )
            V = torch.where(torch.isfinite(V), V, torch.zeros_like(V))

    if logger is not None:
        logger.debug(
            "Torch tiled matvec completed.",
            N=int(N),
            tile_size=int(tile_size),
            device=str(device),
        )
    return V


def _bem_matvec_keops(
    sigma: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    self_integrals: Optional[Tensor],
    logger: Optional[JsonlLogger] = None,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    KeOps-based non-differentiable matvec.

    This path is optional and will raise if KeOps is not available; callers
    are expected to catch and fall back to a Torch implementation.

    Currently only supports the Laplace single-layer kernel.
    """
    if not USE_KEOPS or LazyTensor is None:
        raise RuntimeError("KeOps backend not available.")

    if kernel is not None and not isinstance(kernel, LaplaceSingleLayerKernel):
        raise ValueError(
            "KeOps matvec backend currently only supports LaplaceSingleLayerKernel."
        )

    N = int(sigma.shape[0])
    if N == 0:
        return torch.zeros_like(sigma)

    device = sigma.device

    # KeOps uses (N,1,3) and (1,N,3) layouts for broadcasting.
    X_i = LazyTensor(src_centroids[:, None, :])  # (N,1,3)
    X_j = LazyTensor(src_centroids[None, :, :])  # (1,N,3)
    Sigma_j = LazyTensor(sigma[None, :, None])  # (1,N,1)
    Area_j = LazyTensor(areas[None, :, None])  # (1,N,1)

    # Integer indices for diagonal mask
    idx = torch.arange(N, device=device, dtype=torch.int64)
    I = LazyTensor(idx[:, None, None])
    J = LazyTensor(idx[None, :, None])
    is_diag = (I == J)

    D2_ij = ((X_i - X_j) ** 2).sum(-1)  # (N,N,1)
    eps = 1e-18
    inv_r = (1.0 / (D2_ij + eps).sqrt())
    K_ij = K_E * inv_r  # Laplace kernel

    V = (K_ij * (1 - is_diag) * Sigma_j * Area_j).sum(dim=1).view(N)
    if self_integrals is not None:
        V = V + self_integrals * sigma * areas

    if logger is not None:
        logger.debug("KeOps matvec completed.", N=int(N), device=str(device))
    return V


def _bem_potential_targets_keops(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    logger: Optional[JsonlLogger] = None,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    KeOps-based non-differentiable potential at arbitrary targets.

    Currently only supports the Laplace single-layer kernel.
    """
    if not USE_KEOPS or LazyTensor is None:
        raise RuntimeError("KeOps backend not available.")

    if kernel is not None and not isinstance(kernel, LaplaceSingleLayerKernel):
        raise ValueError(
            "KeOps potential backend currently only supports LaplaceSingleLayerKernel."
        )

    device = targets.device
    dtype = targets.dtype
    M, N = int(targets.shape[0]), int(sigma.shape[0])

    if M == 0 or N == 0:
        return torch.zeros(M, device=device, dtype=dtype)

    # (M,1,3) and (1,N,3) layouts
    X_i = LazyTensor(targets[:, None, :])  # (M,1,3)
    X_j = LazyTensor(src_centroids[None, :, :])  # (1,N,3)
    Sigma_j = LazyTensor(sigma[None, :, None])  # (1,N,1)
    Area_j = LazyTensor(areas[None, :, None])  # (1,N,1)

    D2_ij = ((X_i - X_j) ** 2).sum(-1)  # (M,N,1)
    eps = 1e-18
    inv_r = 1.0 / (D2_ij + eps).sqrt()
    K_ij = K_E * inv_r  # Laplace kernel

    V = (K_ij * Sigma_j * Area_j).sum(dim=1).view(M)

    if logger is not None:
        logger.debug(
            "KeOps potential targets completed.",
            M=int(M),
            N=int(N),
            device=str(device),
        )
    return V


def _bem_E_field_targets_keops(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    logger: Optional[JsonlLogger] = None,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    KeOps-based non-differentiable E-field at arbitrary targets.

    Currently only supports the Laplace single-layer kernel.
    """
    if not USE_KEOPS or LazyTensor is None:
        raise RuntimeError("KeOps backend not available.")

    if kernel is not None and not isinstance(kernel, LaplaceSingleLayerKernel):
        raise ValueError(
            "KeOps E-field backend currently only supports LaplaceSingleLayerKernel."
        )

    device = targets.device
    dtype = targets.dtype
    M, N = int(targets.shape[0]), int(sigma.shape[0])

    if M == 0 or N == 0:
        return torch.zeros(M, 3, device=device, dtype=dtype)

    # (M,1,3) and (1,N,3) layouts
    X_i = LazyTensor(targets[:, None, :])  # (M,1,3)
    X_j = LazyTensor(src_centroids[None, :, :])  # (1,N,3)
    SigmaA_j = LazyTensor((sigma * areas)[None, :, None])  # (1,N,1)

    D_ij = X_i - X_j  # (M,N,3)
    D2_ij = (D_ij ** 2).sum(-1)  # (M,N,1)
    eps = 1e-18
    inv_r = 1.0 / (D2_ij + eps).sqrt()
    inv_r3 = inv_r * inv_r * inv_r  # (M,N,1)

    coef_ij = K_E * SigmaA_j * inv_r3  # (M,N,1)
    E_ij = coef_ij * D_ij  # (M,N,3)
    E = E_ij.sum(dim=1).view(M, 3)

    if logger is not None:
        logger.debug(
            "KeOps E-field targets completed.",
            M=int(M),
            N=int(N),
            device=str(device),
        )
    return E


# --------------------------------------------------------------------------------------
# Public non-diff matvec: GPU-friendly, backend-selectable
# --------------------------------------------------------------------------------------


def bem_matvec_gpu(
    sigma: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    *,
    tile_size: int = 1024,
    self_integrals: Optional[Tensor] = None,
    logger: Optional[JsonlLogger] = None,
    use_keops: bool = False,
    kernel: Optional[SingleLayerKernel] = None,
    backend: str = "auto",
    matvec_impl: Optional[Callable[..., Tensor]] = None,
    **kwargs,
) -> Tensor:
    """
    Matrix-free potential: V = Sum_j K_ij * (sigma_j * area_j)

    Arguments
    ---------
    sigma : (N,) tensor
        Surface charge density on each panel.
    src_centroids : (N,3) tensor
        Panel centroids.
    areas : (N,) tensor
        Panel areas.

    Keyword-only
    ------------
    tile_size : int
        Tile size for Torch-based implementations.
    self_integrals : (N,) tensor or None
        Optional diagonal replacement.
    logger : JsonlLogger or None
        Logger for diagnostics.
    use_keops : bool
        Legacy flag; prefer backend="keops" or backend="auto".
    kernel : SingleLayerKernel or None
        Green's function / physics prior. Defaults to LaplaceSingleLayerKernel.
    backend : {"auto", "torch_tiled", "keops", "external"}
        Backend selection.
    matvec_impl : callable or None
        Custom matvec implementation used when backend="external".
    kwargs :
        - self_correction : alias for self_integrals (for compatibility).
        - use_near_quad : accepted but currently ignored; kept for API stability.

    Notes
    -----
    - This function is intentionally non-differentiable (uses no_grad through
      the internal Torch path).
    - It is safe to use as the A(x) callback for GMRES.
    """
    # compatibility alias for older call sites / tests
    if "self_correction" in kwargs and self_integrals is None:
        self_integrals = kwargs.get("self_correction")

    # Accept but ignore near-field quadrature flag for now
    if "use_near_quad" in kwargs and logger is not None:
        logger.debug(
            "bem_matvec_gpu called with use_near_quad; "
            "near-field quadrature not implemented in this backend."
        )

    if backend not in {"auto", "torch_tiled", "keops", "external"}:
        raise ValueError(f"Unknown BEM backend: {backend!r}")

    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL
    N = int(sigma.shape[0])
    tile_size = _ensure_tile_size(tile_size)

    # Optional external backend hook (e.g., FMM or H-matrix)
    if backend == "external":
        if matvec_impl is None:
            raise ValueError("backend='external' requires matvec_impl.")
        return matvec_impl(
            sigma=sigma,
            src_centroids=src_centroids,
            areas=areas,
            tile_size=tile_size,
            self_integrals=self_integrals,
            logger=logger,
            kernel=ker,
            **kwargs,
        )

    effective_backend = "torch_tiled"

    # Optional KeOps path (auto / explicit)
    if backend in {"auto", "keops"}:
        want_keops = (
            backend == "keops"
            or use_keops
            or os.environ.get("EDE_BEM_USE_KEOPS", "0") == "1"
        )

        if want_keops:
            if not USE_KEOPS:
                if logger is not None:
                    logger.warning(
                        "KeOps requested but not available; falling back to torch tiled."
                    )
            elif N < 2048:
                if logger is not None:
                    logger.debug("N too small for KeOps; using torch tiled.", N=int(N))
            else:
                try:
                    out = _bem_matvec_keops(
                        sigma=sigma,
                        src_centroids=src_centroids,
                        areas=areas,
                        self_integrals=self_integrals,
                        logger=logger,
                        kernel=ker,
                    )
                    effective_backend = "keops"
                    if logger is not None:
                        logger.info(
                            "BEM matvec using KeOps backend.",
                            N=int(N),
                            kernel=ker.name,
                        )
                    return out
                except Exception as exc:  # pragma: no cover - defensive
                    if logger is not None:
                        logger.error(
                            "KeOps matvec failed; falling back to torch tiled.",
                            error=str(exc),
                            exc_info=True,
                        )

    # Torch tiled path
    V = _bem_matvec_torch_tiled(
        sigma=sigma,
        src_centroids=src_centroids,
        areas=areas,
        tile_size=tile_size,
        self_integrals=self_integrals,
        logger=logger,
        kernel=ker,
    )

    if logger is not None:
        logger.info(
            "BEM matvec completed.",
            backend=effective_backend,
            N=int(N),
            tile_size=int(tile_size),
            kernel=ker.name,
        )
    return V


# --------------------------------------------------------------------------------------
# Batched matvec for dictionary building / sparse regression (Pattern 1)
# --------------------------------------------------------------------------------------


def bem_matvec_batched(
    sigma: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    *,
    tile_size: int = 1024,
    self_integrals: Optional[Tensor] = None,
    kernel: Optional[SingleLayerKernel] = None,
    logger: Optional[JsonlLogger] = None,
) -> Tensor:
    """
    Batched matrix-vector apply for dictionary building / sparse regression.

    Parameters
    ----------
    sigma : (N,) or (B, N) tensor
        If 1D, behaves like `bem_matvec_diff`.
        If 2D, B independent sigma vectors are applied in one pass.

    Returns
    -------
    V : (N,) or (B, N) tensor
    """
    if sigma.ndim == 1:
        # Call the differentiable version as this path is for AI/discovery
        return bem_matvec_diff(
            sigma,
            src_centroids,
            areas,
            tile_size=tile_size,
            self_integrals=self_integrals,
            kernel=kernel,
        )

    if sigma.ndim != 2:
        raise ValueError("sigma must have shape (N,) or (B, N).")

    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL
    B, N = sigma.shape
    device, dtype = sigma.device, sigma.dtype

    tile_size = _ensure_tile_size(tile_size)

    V = torch.zeros(B, N, device=device, dtype=dtype)
    if N == 0:
        return V

    assert src_centroids.shape[0] == N, "src_centroids and sigma must have the same length"
    assert areas.shape[0] == N, "areas and sigma must have the same length"

    for i0 in range(0, N, tile_size):
        i1 = min(i0 + tile_size, N)
        chunk = src_centroids[i0:i1]  # (T,3)
        diff = chunk[:, None, :] - src_centroids[None, :, :]  # (T,N,3)
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-12)  # (T,N)
        K = ker.potential(diff, r)  # (T,N)

        # Diagonal indices
        arng = torch.arange(i0, i1, device=device)
        local_idx = torch.arange(i1 - i0, device=device)

        if self_integrals is not None:
            # replace diagonal
            K[local_idx, arng] = self_integrals[arng]
        else:
            # zero diagonal (consistency with FMM/KeOps)
            K[local_idx, arng] = 0.0

        # Broadcast across batch: (1,T,N) * (B,1,N) * (1,1,N) -> (B,T,N)
        contrib = (
            K[None, :, :] * sigma[:, None, :] * areas[None, None, :]
        )
        V[:, i0:i1] = contrib.sum(dim=-1)

    if logger is not None:
        logger.debug(
            "Batched matvec completed.",
            B=int(B),
            N=int(N),
            tile_size=int(tile_size),
            kernel=ker.name,
        )
    return V


# --------------------------------------------------------------------------------------
# Non-diff potentials / fields at targets (GPU aware, KeOps-capable)
# --------------------------------------------------------------------------------------


def bem_potential_targets(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    *,
    logger: Optional[JsonlLogger] = None,
    use_keops: bool = False,
    kernel: Optional[SingleLayerKernel] = None,
    backend: str = "auto",
) -> Tensor:
    """
    Evaluate induced potential at arbitrary targets due to panel charges (non-diff wrapper).

    Backend selection is analogous to `bem_matvec_gpu`.
    """
    if backend not in {"auto", "torch_tiled", "keops"}:
        raise ValueError(f"Unknown backend for bem_potential_targets: {backend!r}")

    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL
    M, N = int(targets.shape[0]), int(sigma.shape[0])
    tile_size = _ensure_tile_size(tile_size)

    effective_backend = "torch_tiled"

    if backend in {"auto", "keops"}:
        want_keops = (
            backend == "keops"
            or use_keops
            or os.environ.get("EDE_BEM_USE_KEOPS", "0") == "1"
        )
        if want_keops:
            if not USE_KEOPS:
                if logger is not None:
                    logger.warning(
                        "KeOps requested for potential targets but not available; "
                        "falling back to torch tiled."
                    )
            else:
                try:
                    out = _bem_potential_targets_keops(
                        targets=targets,
                        src_centroids=src_centroids,
                        areas=areas,
                        sigma=sigma,
                        logger=logger,
                        kernel=ker,
                    )
                    effective_backend = "keops"
                    if logger is not None:
                        logger.info(
                            "BEM potential targets using KeOps backend.",
                            M=int(M),
                            N=int(N),
                            kernel=ker.name,
                        )
                    return out
                except Exception as exc:  # pragma: no cover - defensive
                    if logger is not None:
                        logger.error(
                            "KeOps potential targets failed; falling back to torch tiled.",
                            error=str(exc),
                            exc_info=True,
                        )

    # Torch path
    with torch.no_grad():
        V = _bem_potential_targets_core_torch(
            targets=targets,
            src_centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            tile_size=tile_size,
            kernel=ker,
        )
        if not torch.all(torch.isfinite(V)):
            if logger is not None:
                logger.error(
                    "Non-finite entries detected in bem_potential_targets output; "
                    "clamping to zero.",
                )
            V = torch.where(torch.isfinite(V), V, torch.zeros_like(V))

    if logger is not None:
        logger.info(
            "BEM potential targets completed.",
            backend=effective_backend,
            M=int(M),
            N=int(N),
            tile_size=int(tile_size),
            kernel=ker.name,
        )
    return V


def bem_E_field_targets(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    *,
    logger: Optional[JsonlLogger] = None,
    use_keops: bool = False,
    kernel: Optional[SingleLayerKernel] = None,
    backend: str = "auto",
) -> Tensor:
    """
    Evaluate induced E-field at arbitrary targets due to panel charges (non-diff wrapper).

    Returns
    -------
    Tensor of shape [M, 3]
    """
    if backend not in {"auto", "torch_tiled", "keops"}:
        raise ValueError(f"Unknown backend for bem_E_field_targets: {backend!r}")

    ker = kernel or DEFAULT_SINGLE_LAYER_KERNEL
    M, N = int(targets.shape[0]), int(sigma.shape[0])
    tile_size = _ensure_tile_size(tile_size)

    effective_backend = "torch_tiled"

    if backend in {"auto", "keops"}:
        want_keops = (
            backend == "keops"
            or use_keops
            or os.environ.get("EDE_BEM_USE_KEOPS", "0") == "1"
        )
        if want_keops:
            if not USE_KEOPS:
                if logger is not None:
                    logger.warning(
                        "KeOps requested for E-field targets but not available; "
                        "falling back to torch tiled."
                    )
            else:
                try:
                    out = _bem_E_field_targets_keops(
                        targets=targets,
                        src_centroids=src_centroids,
                        areas=areas,
                        sigma=sigma,
                        logger=logger,
                        kernel=ker,
                    )
                    effective_backend = "keops"
                    if logger is not None:
                        logger.info(
                            "BEM E-field targets using KeOps backend.",
                            M=int(M),
                            N=int(N),
                            kernel=ker.name,
                        )
                    return out
                except Exception as exc:  # pragma: no cover - defensive
                    if logger is not None:
                        logger.error(
                            "KeOps E-field targets failed; falling back to torch tiled.",
                            error=str(exc),
                            exc_info=True,
                        )

    with torch.no_grad():
        E = _bem_E_field_targets_core_torch(
            targets=targets,
            src_centroids=src_centroids,
            areas=areas,
            sigma=sigma,
            tile_size=tile_size,
            kernel=ker,
        )
        if not torch.all(torch.isfinite(E)):
            if logger is not None:
                logger.error(
                    "Non-finite entries detected in bem_E_field_targets output; "
                    "clamping to zero.",
                )
            E = torch.where(torch.isfinite(E), E, torch.zeros_like(E))

    if logger is not None:
        logger.info(
            "BEM E-field targets completed.",
            backend=effective_backend,
            M=int(M),
            N=int(N),
            tile_size=int(tile_size),
            kernel=ker.name,
        )
    return E


# --------------------------------------------------------------------------------------
# Explicit differentiable public APIs (Pattern 3)
# --------------------------------------------------------------------------------------


def bem_matvec_diff(
    sigma: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    *,
    tile_size: int = 1024,
    self_integrals: Optional[Tensor] = None,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable single-layer matvec.

    Use this inside differentiable optimization loops (Pattern 3).
    """
    return _bem_matvec_core_torch(
        centroids=src_centroids,
        areas=areas,
        sigma=sigma,
        self_integrals=self_integrals,
        tile_size=tile_size,
        kernel=kernel,
    )


def bem_potential_targets_diff(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable induced potential at arbitrary targets.
    """
    return _bem_potential_targets_core_torch(
        targets=targets,
        src_centroids=src_centroids,
        areas=areas,
        sigma=sigma,
        tile_size=tile_size,
        kernel=kernel,
    )


def bem_E_field_targets_diff(
    targets: Tensor,
    src_centroids: Tensor,
    areas: Tensor,
    sigma: Tensor,
    tile_size: int = 1024,
    kernel: Optional[SingleLayerKernel] = None,
) -> Tensor:
    """
    Differentiable induced E-field at arbitrary targets.
    """
    return _bem_E_field_targets_core_torch(
        targets=targets,
        src_centroids=src_centroids,
        areas=areas,
        sigma=sigma,
        tile_size=tile_size,
        kernel=kernel,
    )


__all__ = [
    # core kernels
    "_bem_matvec_core_torch",
    "_bem_potential_targets_core_torch",
    "_bem_E_field_targets_core_torch",
    # non-diff APIs
    "bem_matvec_gpu",
    "bem_matvec_batched",
    "bem_potential_targets",
    "bem_E_field_targets",
    # diff APIs
    "bem_matvec_diff",
    "bem_potential_targets_diff",
    "bem_E_field_targets_diff",
    # kernel abstractions
    "SingleLayerKernel",
    "LaplaceSingleLayerKernel",
    "DEFAULT_SINGLE_LAYER_KERNEL",
    "have_keops",
]