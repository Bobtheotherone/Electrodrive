# bem_fmm.py
from __future__ import annotations

"""
High-level BEM ⇄ FMM glue for the Laplace single-layer operator.

This module implements a *numerically robust* external backend for
:func:`electrodrive.core.bem_kernel.bem_matvec_gpu` based on the Tier-3
FMM stack in :mod:`electrodrive.fmm3d`.

Compared to the original prototype treecode implementation, this version:

- delegates geometry + interaction logic to :mod:`electrodrive.fmm3d`,
- uses high-order spherical-harmonic expansions (order p >= 4),
- cleanly separates near-field (exact P2P) and far-field (M2L/L2P),
- enforces consistent physics with the BEM kernel (same K_E, same
  handling of diagonal self-integrals),
- exposes instrumentation hooks via :class:`MultipoleOpStats` and the
  standard :class:`JsonlLogger` used elsewhere in the project`.

Public API
==========

The main entry points are

- :class:`LaplaceFmm3D` – stateful backend object owning the tree and FMM
  configuration, with a :meth:`matvec` method compatible with
  ``bem_matvec_gpu(backend="external")``.
- :func:`make_laplace_fmm_backend` – convenience constructor used by
  tests and orchestration helpers.
"""

from dataclasses import dataclass
from typing import Optional, Any

import copy
import torch
from torch import Tensor

from electrodrive.utils.config import K_E
from electrodrive.fmm3d.config import FmmConfig, BackendKind
from electrodrive.utils.logging import JsonlLogger
from electrodrive.core.bem_kernel import (
    LaplaceSingleLayerKernel,
    SingleLayerKernel,
    DEFAULT_SINGLE_LAYER_KERNEL,
)

from electrodrive.fmm3d.tree import FmmTree, build_fmm_tree
from electrodrive.fmm3d.interaction_lists import (
    InteractionLists,
    build_interaction_lists,
)
from electrodrive.fmm3d.multipole_operators import (
    MultipoleCoefficients,
    LocalCoefficients,
    MultipoleOpStats,
)
from electrodrive.fmm3d.kernels_cpu import (
    apply_p2p_cpu,
    apply_p2p_cpu_tiled,  # kept for potential future use / compatibility
    p2m_cpu,
    m2m_cpu,
    m2l_cpu,
    l2l_cpu,
    l2p_cpu,
    P2PResult,
)
from electrodrive.fmm3d.kernels_gpu import (
    apply_p2p_gpu,
    p2m_gpu,
    m2m_gpu,
    m2l_gpu,
    l2l_gpu,
    l2p_gpu,
)
from electrodrive.fmm3d.logging_utils import get_logger


__all__ = ["LaplaceFmm3D", "make_laplace_fmm_backend"]


@dataclass
class _BemFmmState:
    """
    Internal cached state for a symmetric BEM FMM matvec.

    Attributes
    ----------
    tree:
        Geometry tree over the panel centroids (sources == targets).
    cfg:
        FMM configuration (expansion order, MAC, leaf size, dtype).
    lists:
        Interaction lists used for near-field P2P (U-list) and far-field
        M2L (V/W/X lists).
    """

    tree: FmmTree
    cfg: FmmConfig
    lists: InteractionLists


class LaplaceFmm3D:
    """
    High-order 3D Laplace FMM backend for the single-layer BEM operator.

    This class is intentionally minimal: it owns a geometry tree and an
    :class:`FmmConfig` instance and exposes a single :meth:`matvec`
    method that matches the external-backend hook of
    :func:`bem_matvec_gpu`.

    Notes
    -----
    - Geometry passed to the constructor (``src_centroids`` and
      ``areas``) must live on the CPU. Tree construction and interaction
      list generation are CPU-only for correctness and debuggability.
      Heavy FMM work (multipole translations + P2P) can run either on
      the CPU or on a CUDA device depending on ``backend``.
    - The expansion order is controlled via :class:`FmmConfig` and
      defaults to whatever that class uses (currently p = 8).  This is
      typically sufficient to achieve < 1% relative error on well-behaved
      meshes when combined with a reasonably strict MAC (e.g.
      ``theta ≲ 0.6``).
    """

    def __init__(
        self,
        src_centroids: Tensor,
        areas: Tensor,
        *,
        max_leaf_size: int = 64,
        theta: float = 0.5,
        use_dipole: bool = True,
        logger: Optional[Any] = None,
        expansion_order: Optional[int] = None,
        backend: BackendKind = "auto",
        device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Parameters
        ----------
        src_centroids : (N, 3) tensor
            Panel centroids (sources == targets for BEM matvecs).
        areas : (N,) tensor
            Panel areas. Used only to form physical charges
            ``q = sigma * area`` per matvec.
        max_leaf_size : int, optional
            Maximum number of points per leaf node in the FMM tree.
        theta : float, optional
            MAC (multipole-acceptance criterion) parameter.  Smaller
            values yield more accurate but more expensive FMMs.
        use_dipole : bool, optional
            Kept for backwards compatibility with the original treecode
            backend; has no effect in the high-order FMM implementation.
        logger : JsonlLogger or None
            Optional structured logger for diagnostics.
        expansion_order : int or None, optional
            Override the FMM expansion order ``p``.  If None, the
            default from :class:`FmmConfig` is used (currently p = 8).
        backend : {"cpu", "gpu", "auto"}, optional
            Logical backend selector. "auto" chooses a GPU backend when
            available and allowed by ``FmmConfig.use_gpu``, otherwise
            falls back to CPU.
        device : torch.device or str or None, optional
            Preferred device for FMM work when ``backend`` resolves to
            "gpu". If ``None``, defaults to the current CUDA device.
        """
        if src_centroids.ndim != 2 or src_centroids.shape[1] != 3:
            raise ValueError("src_centroids must have shape (N, 3)")
        if areas.ndim != 1 or areas.shape[0] != src_centroids.shape[0]:
            raise ValueError("areas must have shape (N,) matching src_centroids")

        # The constructor remains CPU-only: geometry tensors must live on
        # the CPU. GPU acceleration is handled internally by moving the
        # *tree* and per-matvec charges, not by accepting CUDA geometry.
        if src_centroids.device.type != "cpu" or areas.device.type != "cpu":
            raise ValueError(
                "LaplaceFmm3D currently supports only CPU tensors for "
                "src_centroids and areas."
            )

        self.src_centroids = src_centroids
        self.areas = areas
        self.N = int(src_centroids.shape[0])
        self.max_leaf_size = int(max_leaf_size)
        self.theta = float(theta)
        self.use_dipole = bool(use_dipole)  # retained for API stability

        # Normalize logger: supports fan-out to console if verbose mode is enabled.
        self.logger = get_logger(logger)

        self.device = src_centroids.device  # CPU device for public API
        self.dtype = src_centroids.dtype

        # Build FMM configuration.  We let FmmConfig handle validation.
        # dtype is derived from src_centroids.dtype so that multipole math
        # stays numerically consistent with the BEM layer.
        cfg_kwargs = dict(
            mac_theta=self.theta,
            leaf_size=self.max_leaf_size,
            dtype=self.dtype,
            backend=backend,
        )
        if expansion_order is not None:
            cfg_kwargs["expansion_order"] = int(expansion_order)

        self.cfg = FmmConfig(**cfg_kwargs)

        # Resolve backend and FMM device after config construction so that
        # we can respect cfg.use_gpu and keep dtype unchanged.
        if backend == "auto":
            if torch.cuda.is_available() and self.cfg.use_gpu:
                self.backend: BackendKind = "gpu"
                self.fmm_device = torch.device(device or "cuda")
            else:
                self.backend = "cpu"
                self.fmm_device = self.src_centroids.device
        elif backend == "gpu":
            if not torch.cuda.is_available() or not self.cfg.use_gpu:
                raise RuntimeError("GPU backend requested but CUDA/use_gpu not available.")
            self.backend = "gpu"
            self.fmm_device = torch.device(device or "cuda")
        else:
            # Force CPU execution even if GPUs are available.
            self.backend = "cpu"
            self.fmm_device = self.src_centroids.device

        # Keep the config in sync with the resolved backend.
        self.cfg.backend = self.backend

        # Build geometry tree on CPU in original point ordering.
        # The tree stores points in its own "tree order"; we keep the
        # original centroids separately for pointer sanity checks.
        if self.N > 0:
            # IMPORTANT: respect cfg.leaf_size so that the geometry
            # driving the interaction lists matches the FMM config.
            tree = build_fmm_tree(
                self.src_centroids,
                leaf_size=int(self.cfg.leaf_size),
            )
            # Precompute interaction lists once, based on MAC.
            lists = build_interaction_lists(
                tree,
                tree,
                mac_theta=self.cfg.mac_theta,
            )
        else:
            # Degenerate empty tree; still construct a minimal FmmTree
            # via build_fmm_tree to keep invariants consistent.
            tree = build_fmm_tree(
                torch.zeros(0, 3, dtype=self.dtype, device=self.device),
                leaf_size=int(self.cfg.leaf_size),
            )
            lists = build_interaction_lists(
                tree,
                tree,
                mac_theta=self.cfg.mac_theta,
            )

        # CPU-oriented cached state.
        self.tree_cpu: FmmTree = tree
        self.lists: InteractionLists = lists
        self.state = _BemFmmState(tree=self.tree_cpu, cfg=self.cfg, lists=self.lists)
        # Backwards-compatibility alias for existing code that expects `backend.tree`.
        self.tree: FmmTree = self.tree_cpu

        # ------------------------------------------------------------------
        # Scratch buffers and GPU geometry caches
        # ------------------------------------------------------------------
        # Reusable host-side buffer for physical charges q = sigma * area.
        self._q_host: Optional[Tensor] = None

        # Optional pinned host buffer to accelerate host→device copies
        # in the GPU backend. Allocation is best-effort.
        self._q_host_pinned: Optional[Tensor] = None

        # Optional GPU-side copy of panel areas so we can form q on device.
        self._areas_gpu: Optional[Tensor] = None

        if self.N > 0:
            self._q_host = torch.empty(self.N, dtype=self.dtype, device=self.device)

        # Optional GPU clone of the tree, created eagerly for GPU backends.
        self.tree_gpu: Optional[FmmTree] = None
        if self.backend == "gpu":
            self.tree_gpu = copy.deepcopy(self.tree_cpu)
            self.tree_gpu.to(self.fmm_device, dtype=self.cfg.dtype)

            if self.N > 0:
                # Best-effort pinned scratch buffer.
                try:
                    self._q_host_pinned = torch.empty(
                        self.N, dtype=self.dtype, pin_memory=True
                    )
                except RuntimeError:
                    self._q_host_pinned = None

                # Persistent GPU copy of areas for forming q directly on device.
                self._areas_gpu = self.areas.to(
                    device=self.fmm_device, dtype=self.cfg.dtype
                )

        if self.logger is not None:
            self.logger.info(
                "LaplaceFmm3D backend constructed.",
                N=int(self.N),
                leaf_size=int(self.max_leaf_size),
                theta=float(self.theta),
                expansion_order=int(self.cfg.expansion_order),
                dtype=str(self.dtype),
                backend=self.backend,
                fmm_device=str(self.fmm_device),
            )

    # ------------------------------------------------------------------
    # Public / internal helpers
    # ------------------------------------------------------------------

    def _check_geometry_consistency(
        self,
        src_centroids: Tensor,
        areas: Tensor,
    ) -> None:
        """
        Basic sanity checks that the matvec call matches the geometry
        used to build the backend.
        """
        if src_centroids.data_ptr() != self.src_centroids.data_ptr():
            raise ValueError(
                "LaplaceFmm3D.matvec called with different src_centroids "
                "than the ones used to build the FMM tree."
            )
        if areas.data_ptr() != self.areas.data_ptr():
            raise ValueError(
                "LaplaceFmm3D.matvec called with different areas "
                "than the ones used to build the FMM tree."
            )
        if src_centroids.device.type != "cpu" or areas.device.type != "cpu":
            raise ValueError(
                "LaplaceFmm3D currently requires matvec geometry tensors "
                "to live on the CPU."
            )
        if src_centroids.dtype != self.dtype or areas.dtype != self.dtype:
            raise ValueError(
                "LaplaceFmm3D geometry dtype mismatch: "
                f"expected {self.dtype}, got "
                f"{src_centroids.dtype} / {areas.dtype}."
            )

    def _maybe_update_p2p_batch_size(self, tile_size: int) -> None:
        """
        Update cfg.p2p_batch_size only when it actually changes.

        This keeps the semantics identical while avoiding unnecessary
        attribute writes on hot matvec paths.
        """
        if hasattr(self.state.cfg, "p2p_batch_size"):
            tile_size_int = int(tile_size)
            if getattr(self.state.cfg, "p2p_batch_size", None) != tile_size_int:
                self.state.cfg.p2p_batch_size = tile_size_int

    def _record_stage_norms(
        self,
        *,
        logger: Optional[Any],
        stats: MultipoleOpStats,
        phi_far: Tensor,
        phi_near: Tensor,
    ) -> None:
        """
        Best-effort logging of ||phi_far||_2 and ||phi_near||_2.

        This is gated on logger != None to avoid doing extra work on
        production runs where instrumentation is disabled.
        """
        if logger is None:
            return
        try:
            stats.extras["l2_norm_phi_far"] = float(
                torch.linalg.vector_norm(phi_far).item()
            )
            stats.extras["l2_norm_phi_near"] = float(
                torch.linalg.vector_norm(phi_near).item()
            )
        except Exception:
            # Telemetry must never interfere with the numerical path.
            pass

    def _log_stats(
        self,
        logger: Optional[Any],
        stats: MultipoleOpStats,
        p2p_result: Optional[P2PResult] = None,
    ) -> None:
        """
        Emit a structured debug log with FMM operator statistics.

        The core counters come from :meth:`MultipoleOpStats.as_dict` (if available);
        any additional numeric counters registered in ``stats.extras`` are included
        as well.  P2P-specific metrics (if exposed by ``apply_p2p_cpu_tiled``) are
        logged under a ``p2p_*`` prefix.

        Logging is strictly best-effort: any exception is caught and
        ignored so that instrumentation can never affect the numerical
        path.
        """
        if logger is None:
            return

        try:
            payload = {}

            # Core + extra counters from the multipole backend.
            if hasattr(stats, "as_dict"):
                # Newer MultipoleOpStats implementations
                for k, v in stats.as_dict().items():  # type: ignore[attr-defined]
                    payload[k] = float(v)
            else:
                # Fallback for older MultipoleOpStats without as_dict()
                payload.update(
                    p2m_calls=float(getattr(stats, "p2m_calls", 0)),
                    m2m_calls=float(getattr(stats, "m2m_calls", 0)),
                    m2l_calls=float(getattr(stats, "m2l_calls", 0)),
                    l2l_calls=float(getattr(stats, "l2l_calls", 0)),
                    l2p_calls=float(getattr(stats, "l2p_calls", 0)),
                )
                for k, v in getattr(stats, "extras", {}).items():
                    payload[k] = float(v)

            # Attach a minimal geometry/config snapshot so that operator
            # counts can be correlated with problem size in logs.
            payload.update(
                N=int(self.N),
                n_nodes=int(self.state.tree.n_nodes),
                leaf_size=int(self.max_leaf_size),
                expansion_order=int(self.cfg.expansion_order),
                mac_theta=float(self.cfg.mac_theta),
            )

            # P2P metrics (if the P2P kernel exposes them).
            if p2p_result is not None:
                for attr in ("n_pairs", "n_interactions", "n_tiles"):
                    if hasattr(p2p_result, attr):
                        try:
                            payload[f"p2p_{attr}"] = int(getattr(p2p_result, attr))
                        except Exception:
                            # Best-effort: ignore if non-int-convertible.
                            pass

            logger.debug("LaplaceFmm3D statistics.", **payload)  # type: ignore[arg-type]
        except Exception:
            # Never let instrumentation break the numerical path.
            pass

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _matvec_cpu(
        self,
        *,
        sigma: Tensor,
        tile_size: int,
        self_integrals: Optional[Tensor],
        logger: Optional[Any],
        kernel: Optional[SingleLayerKernel],
    ) -> Tensor:
        """
        CPU implementation of the symmetric Laplace single-layer matvec.

        This is the original matvec body, factored out so that the public
        :meth:`matvec` method can dispatch between CPU and GPU backends.
        """
        del kernel  # unused in the CPU path; kept for signature symmetry

        N = int(sigma.shape[0])
        if N == 0:
            return torch.zeros_like(sigma)

        with torch.no_grad():
            # Physical charges q_j = sigma_j * A_j.
            # Reuse a preallocated host-side buffer when available to
            # avoid allocator churn inside hot matvec loops.
            if self._q_host is not None:
                q = self._q_host
                q.copy_(sigma)
                q.mul_(self.areas)
            else:
                q = sigma * self.areas  # (N,)

            # Upward pass: P2M + M2M on the pre-built tree.
            stats = MultipoleOpStats()
            multipoles: MultipoleCoefficients = p2m_cpu(
                tree=self.state.tree,
                charges=q,
                cfg=self.state.cfg,
                stats=stats,
            )
            multipoles = m2m_cpu(
                tree=self.state.tree,
                multipoles=multipoles,
                cfg=self.state.cfg,
                stats=stats,
            )

            # Far-field: M2L + L2L + L2P → potentials in tree order
            # for the 1/|r| kernel.
            locals_: LocalCoefficients = m2l_cpu(
                source_tree=self.state.tree,
                target_tree=self.state.tree,
                multipoles=multipoles,
                lists=self.state.lists,
                cfg=self.state.cfg,
                stats=stats,
            )
            locals_ = l2l_cpu(
                tree=self.state.tree,
                locals_=locals_,
                cfg=self.state.cfg,
                stats=stats,
            )
            phi_far_tree: Tensor = l2p_cpu(
                tree=self.state.tree,
                locals_=locals_,
                cfg=self.state.cfg,
                stats=stats,
            )

            # Near-field: exact P2P on leaves using precomputed lists.
            # The P2P kernel implements the same 1/|r| kernel as the
            # far-field; the Coulomb constant K_E is applied once to the
            # combined near- and far-field contributions below.
            # charges_src must be in tree order; p2m_cpu already
            # produced multipoles with a tree-ordered charges vector.

            # Allow the user-supplied tile_size to override the config-driven
            # batch size for this call, if that field exists on FmmConfig.
            self._maybe_update_p2p_batch_size(tile_size)

            p2p_result: P2PResult = apply_p2p_cpu(
                source_tree=self.state.tree,
                target_tree=self.state.tree,
                charges_src=multipoles.charges,
                lists=self.state.lists,
                cfg=self.state.cfg,
                logger=logger,  # normalized logger
                out=None,
            )
            phi_p2p_tree: Tensor = p2p_result.potential

            # Stage-wise diagnostics: L2 norms of far-field and near-field
            # contributions for the underlying 1/|r| kernel before
            # applying the physical Coulomb constant.
            self._record_stage_norms(
                logger=logger,
                stats=stats,
                phi_far=phi_far_tree,
                phi_near=phi_p2p_tree,
            )

            # Combine near + far for the 1/|r| kernel.
            phi_total_tree = phi_far_tree + phi_p2p_tree

            # Scale by Coulomb constant so that the overall kernel
            # matches G(r) = K_E / |r| used by the BEM layer.
            phi_total_tree = phi_total_tree * float(K_E)

            # Map back to original panel order.
            V = self.state.tree.map_to_original_order(phi_total_tree)

            # Add diagonal self-integral term if provided:
            #   V_i += self_integrals[i] * sigma_i * area_i
            if self_integrals is not None:
                if self_integrals.shape != sigma.shape:
                    raise ValueError(
                        "self_integrals must have shape (N,) matching sigma."
                    )
                V = V + self_integrals * sigma * self.areas

            # Basic sanity: clamp non-finite results (should not happen).
            mask_finite = torch.isfinite(V)
            if not torch.all(mask_finite):
                if logger is not None:
                    logger.error(
                        "Non-finite entries detected in LaplaceFmm3D matvec "
                        "output; clamping to zero."
                    )
                V = torch.where(mask_finite, V, torch.zeros_like(V))

            # Instrumentation / telemetry
            self._log_stats(logger, stats, p2p_result=p2p_result)

        return V

    def _matvec_gpu(
        self,
        *,
        sigma: Tensor,
        tile_size: int,
        self_integrals: Optional[Tensor],
        logger: Optional[Any],
        kernel: Optional[SingleLayerKernel],
    ) -> Tensor:
        """
        GPU-accelerated implementation of the symmetric Laplace single-layer matvec.

        Heavy FMM work (P2M/M2M/M2L/L2L/L2P + P2P) is carried out on
        ``self.fmm_device`` while the public API remains CPU-oriented.
        """
        del kernel  # unused in the GPU path; kept for signature symmetry

        N = int(sigma.shape[0])
        if N == 0:
            return torch.zeros_like(sigma)

        if self.backend != "gpu":
            raise RuntimeError("LaplaceFmm3D._matvec_gpu called but backend is not 'gpu'.")

        # Lazily build the GPU clone of the tree if it does not exist yet.
        if self.tree_gpu is None:
            self.tree_gpu = copy.deepcopy(self.tree_cpu)
            self.tree_gpu.to(self.fmm_device, dtype=self.cfg.dtype)

        tree_gpu = self.tree_gpu
        assert tree_gpu is not None  # for type checkers

        dev = self.fmm_device
        dtype = self.cfg.dtype
        sigma_device = sigma.device

        with torch.no_grad():
            # ------------------------------------------------------------------
            # Form physical charges q_j = sigma_j * A_j.
            #
            # Preferred path: if we have a GPU copy of the areas, move sigma
            # once and scale in-place on device. Otherwise, fall back to a
            # pinned host buffer (if available) or a plain host buffer.
            # ------------------------------------------------------------------
            if self._areas_gpu is None and self.N > 0:
                self._areas_gpu = self.areas.to(device=dev, dtype=dtype)

            if self._areas_gpu is not None:
                if sigma_device == dev and sigma.dtype == dtype:
                    sigma_dev = sigma
                else:
                    sigma_dev = sigma.to(device=dev, dtype=dtype, non_blocking=True)
                q_gpu = sigma_dev * self._areas_gpu  # q_j on device
            else:
                if self._q_host_pinned is not None:
                    q_cpu = self._q_host_pinned
                    q_cpu.copy_(sigma)
                    q_cpu.mul_(self.areas)
                    q_gpu = q_cpu.to(device=dev, dtype=dtype, non_blocking=True)
                else:
                    q_cpu = sigma * self.areas
                    q_gpu = q_cpu.to(device=dev, dtype=dtype)

            stats = MultipoleOpStats()
            multipoles: MultipoleCoefficients = p2m_gpu(
                tree=tree_gpu,
                charges=q_gpu,
                cfg=self.state.cfg,
                stats=stats,
            )
            multipoles = m2m_gpu(
                tree=tree_gpu,
                multipoles=multipoles,
                cfg=self.state.cfg,
                stats=stats,
            )

            # Far-field on GPU.
            locals_: LocalCoefficients = m2l_gpu(
                source_tree=tree_gpu,
                target_tree=tree_gpu,
                multipoles=multipoles,
                lists=self.state.lists,
                cfg=self.state.cfg,
                stats=stats,
            )
            locals_ = l2l_gpu(
                tree=tree_gpu,
                locals_=locals_,
                cfg=self.state.cfg,
                stats=stats,
            )
            phi_far_tree_gpu: Tensor = l2p_gpu(
                tree=tree_gpu,
                locals_=locals_,
                cfg=self.state.cfg,
                stats=stats,
            )

            # Near-field P2P on GPU. Respect the per-call tile size override.
            self._maybe_update_p2p_batch_size(tile_size)

            p2p_result: P2PResult = apply_p2p_gpu(
                source_tree=tree_gpu,
                target_tree=tree_gpu,
                charges_src=multipoles.charges,
                lists=self.state.lists,
                cfg=self.state.cfg,
                logger=logger,
                out=None,
            )
            phi_p2p_tree_gpu: Tensor = p2p_result.potential

            # L2 diagnostics on GPU contributions.
            self._record_stage_norms(
                logger=logger,
                stats=stats,
                phi_far=phi_far_tree_gpu,
                phi_near=phi_p2p_tree_gpu,
            )

            # Combine near + far (still pure 1/|r|) and scale by K_E.
            phi_total_tree_gpu = phi_far_tree_gpu + phi_p2p_tree_gpu
            phi_total_tree_gpu = phi_total_tree_gpu * float(K_E)

            # Map back to original panel order on GPU, then move to output device.
            V_gpu_orig = tree_gpu.map_to_original_order(phi_total_tree_gpu)
            if sigma_device.type == "cuda":
                V = V_gpu_orig
            else:
                V = V_gpu_orig.to(device=self.device, dtype=self.dtype)

            # Apply diagonal self-integral correction on the output device.
            if self_integrals is not None:
                if self_integrals.shape != sigma.shape:
                    raise ValueError(
                        "self_integrals must have shape (N,) matching sigma."
                    )
                if sigma_device.type == "cuda":
                    if self._areas_gpu is None:
                        self._areas_gpu = self.areas.to(device=dev, dtype=dtype)
                    V = V + self_integrals * sigma * self._areas_gpu
                else:
                    V = V + self_integrals * sigma * self.areas

            # Clamp non-finite entries on CPU.
            mask_finite = torch.isfinite(V)
            if not torch.all(mask_finite):
                if logger is not None:
                    logger.error(
                        "Non-finite entries detected in LaplaceFmm3D matvec "
                        "output; clamping to zero."
                    )
                V = torch.where(mask_finite, V, torch.zeros_like(V))

            # Instrumentation / telemetry
            self._log_stats(logger, stats, p2p_result=p2p_result)

        return V

    # ------------------------------------------------------------------
    # Public matvec API (bem_matvec_gpu external backend)
    # ------------------------------------------------------------------

    def matvec(
        self,
        *,
        sigma: Tensor,
        src_centroids: Tensor,
        areas: Tensor,
        tile_size: int,
        self_integrals: Optional[Tensor],
        logger: Optional[Any] = None,
        kernel: Optional[SingleLayerKernel] = None,
        **kwargs,
    ) -> Tensor:
        """
        External matvec implementation compatible with :func:`bem_matvec_gpu`.

        Parameters are keyword-only to match the call pattern used by
        ``bem_matvec_gpu(backend="external", matvec_impl=fmm.matvec, ...)``.
        """
        del kwargs  # unused, kept for forward-compatibility

        # Geometry consistency (same panel mesh as during construction)
        self._check_geometry_consistency(src_centroids, areas)

        if kernel is not None and not isinstance(kernel, LaplaceSingleLayerKernel):
            raise ValueError(
                "LaplaceFmm3D currently only supports LaplaceSingleLayerKernel."
            )

        N = int(sigma.shape[0])
        if N != self.N:
            raise ValueError(
                f"sigma has length {N}, but FMM backend was built for {self.N} panels."
            )
        sigma_device = sigma.device
        if sigma_device.type == "cuda":
            if self.backend != "gpu":
                raise ValueError(
                    "LaplaceFmm3D.matvec requires backend='gpu' for CUDA sigma. "
                    "Rebuild the backend with backend='gpu' or move sigma to CPU."
                )
            fmm_device = self.fmm_device
            if fmm_device.type == "cuda" and fmm_device.index is None:
                fmm_device = torch.device("cuda", torch.cuda.current_device())
            if sigma_device != fmm_device:
                raise ValueError(
                    "LaplaceFmm3D.matvec CUDA sigma must live on "
                    f"{fmm_device}, got {sigma_device}."
                )
        elif sigma_device.type != "cpu":
            raise ValueError("LaplaceFmm3D.matvec only supports CPU or CUDA sigma.")
        if sigma.dtype != self.dtype:
            raise ValueError(
                f"LaplaceFmm3D sigma dtype mismatch: expected {self.dtype}, "
                f"got {sigma.dtype}."
            )
        if self_integrals is not None:
            if self_integrals.shape != sigma.shape:
                raise ValueError("self_integrals must have shape (N,) matching sigma.")
            if self_integrals.device != sigma_device:
                raise ValueError(
                    "self_integrals must be on the same device as sigma "
                    f"(sigma={sigma_device}, self_integrals={self_integrals.device})."
                )
            if self_integrals.dtype != sigma.dtype:
                raise ValueError(
                    "self_integrals must have the same dtype as sigma "
                    f"(sigma={sigma.dtype}, self_integrals={self_integrals.dtype})."
                )

        # Cheap fast-path fix: ensure we are working with contiguous
        # 1D storage for sigma. We treat sigma as read-only.
        if not sigma.is_contiguous():
            sigma = sigma.contiguous()

        # Resolve logger for this specific call.
        # If none provided, default to the instance logger (which is already normalized).
        # If provided, normalize it to ensure verbose output works for this call too.
        if logger is None:
            logger = self.logger
        else:
            logger = get_logger(logger)

        if logger is not None:
            logger.debug(
                "LaplaceFmm3D matvec called.",
                N=int(N),
                leaf_size=int(self.max_leaf_size),
                theta=float(self.theta),
                expansion_order=int(self.cfg.expansion_order),
                backend=self.backend,
                fmm_device=str(self.fmm_device),
            )

        if N == 0:
            return torch.zeros_like(sigma)

        # Dispatch to backend-specific implementation.
        if self.backend == "gpu":
            V = self._matvec_gpu(
                sigma=sigma,
                tile_size=tile_size,
                self_integrals=self_integrals,
                logger=logger,
                kernel=kernel,
            )
        else:
            V = self._matvec_cpu(
                sigma=sigma,
                tile_size=tile_size,
                self_integrals=self_integrals,
                logger=logger,
                kernel=kernel,
            )

        if logger is not None:
            logger.info(
                "LaplaceFmm3D matvec completed.",
                N=int(N),
                leaf_size=int(self.max_leaf_size),
                theta=float(self.theta),
                expansion_order=int(self.cfg.expansion_order),
                backend=self.backend,
                fmm_device=str(self.fmm_device),
            )

        return V


def make_laplace_fmm_backend(
    src_centroids: Tensor,
    areas: Tensor,
    *,
    max_leaf_size: int = 64,
    theta: float = 0.5,
    use_dipole: bool = True,
    logger: Optional[Any] = None,
    expansion_order: Optional[int] = None,
    backend: BackendKind = "auto",
    device: Optional[torch.device | str] = None,
) -> LaplaceFmm3D:
    """
    Convenience constructor for a Laplace FMM backend.

    Typical usage
    -------------
    >>> fmm = make_laplace_fmm_backend(centroids, areas)
    >>> V = bem_matvec_gpu(
    ...     sigma,
    ...     centroids,
    ...     areas,
    ...     backend="external",
    ...     matvec_impl=fmm.matvec,
    ...     kernel=DEFAULT_SINGLE_LAYER_KERNEL,
    ... )

    Parameters
    ----------
    src_centroids, areas, max_leaf_size, theta, use_dipole, logger, expansion_order
        Passed through to :class:`LaplaceFmm3D`.
    backend :
        Logical backend selector ("cpu", "gpu", or "auto").
    device :
        Optional device hint for GPU backends.

    This uses the same Laplace single-layer kernel as the default
    Torch/KeOps path, but accelerates the matvec via a high-order FMM
    approximation.  On modest expansion orders (e.g. p ≈ 8) and
    MAC parameters around theta ≈ 0.5–0.6, relative errors of
    ``O(10^{-2})`` or better are typically achievable on realistic meshes.
    """
    return LaplaceFmm3D(
        src_centroids=src_centroids,
        areas=areas,
        max_leaf_size=max_leaf_size,
        theta=theta,
        use_dipole=use_dipole,
        logger=logger,
        expansion_order=expansion_order,
        backend=backend,
        device=device,
    )
