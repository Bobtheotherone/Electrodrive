from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from electrodrive.utils.config import K_E
from .config import FmmConfig
from .tree import build_fmm_tree
from .interaction_lists import build_interaction_lists
from .multipole_operators import p2m, m2m, m2l, l2l, l2p
from .kernels_cpu import apply_p2p_cpu_tiled
from .logging_utils import log_fmm_event

try:
    # Optional: existing treecode-style backend, if present.
    from electrodrive.core import bem_fmm as legacy_bem_fmm  # type: ignore
except Exception:  # pragma: no cover
    legacy_bem_fmm = None  # type: ignore


@dataclass
class FmmBemBackend:
    """Thin wrapper used by BEM to call into the FMM stack.

    The main entry point is ``apply(src_centroids, sigma)`` which matches
    the interface expected by ``bem_matvec_gpu``'s ``matvec_impl`` hook.
    """

    cfg: FmmConfig
    logger: Optional[object] = None

    def apply(
        self,
        src_centroids: Tensor,
        sigma: Tensor,
        areas: Optional[Tensor] = None,
        *,
        self_integrals: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Apply the FMM-accelerated single-layer potential.

        This implements the full FMM pipeline (Near + Far field) to ensure
        numerical correctness.

        Parameters
        ----------
        src_centroids : (N, 3) tensor
            Panel centroids.
        sigma : (N,) tensor
            Surface charge densities.
        areas : (N,) tensor or None
            Panel areas. If None, they are assumed to be 1.
        self_integrals : (N,) tensor or None
            Optional self-interaction correction terms for the diagonal.

        Returns
        -------
        Tensor
            Approximated potential at the panel centroids.
        """
        self.cfg.validate()
        N = int(sigma.shape[0])
        device = sigma.device
        if areas is None:
            areas = torch.ones_like(sigma)

        log_fmm_event(
            self.logger,
            "fmm_bem_apply_start",
            N=N,
            backend="tier3_fmm_scaffold",
            expansion_order=int(self.cfg.expansion_order),
        )

        # Placeholder path: fall back to legacy backend if available.
        if legacy_bem_fmm is not None and hasattr(legacy_bem_fmm, "bem_fmm_matvec"):
            V = legacy_bem_fmm.bem_fmm_matvec(src_centroids, sigma, areas)  # type: ignore[attr-defined]
        else:
            # Tier-3 FMM execution
            tree = build_fmm_tree(src_centroids, leaf_size=self.cfg.leaf_size)
            lists = build_interaction_lists(tree, tree, mac_theta=self.cfg.mac_theta)

            # Physical charges q_j = sigma_j * A_j for the 1/|r| kernel.
            charges = sigma * areas

            # 1. Far-field (Spectral)
            # p2m returns multipoles with charges mapped to tree order.
            multipoles = p2m(tree, charges, self.cfg)
            multipoles = m2m(tree, multipoles, self.cfg)
            locals_ = m2l(tree, tree, multipoles, self.cfg)
            locals_ = l2l(tree, locals_, self.cfg)
            phi_far_tree = l2p(tree, locals_, self.cfg)

            # 2. Near-field (P2P)
            # apply_p2p_cpu_tiled expects charges in tree order.
            # multipoles.charges holds exactly that.
            p2p_res = apply_p2p_cpu_tiled(
                source_tree=tree,
                target_tree=tree,
                charges_src=multipoles.charges,
                lists=lists,
                tile_size_points=1024,  # sensible default for CPU
                logger=self.logger,
            )
            phi_near_tree = p2p_res.potential

            # 3. Combine and Scale
            # Both kernels are 1/|r|; apply Coulomb constant here.
            phi_total_tree = (phi_far_tree + phi_near_tree) * float(K_E)

            # 4. Map back to original order
            V = tree.map_to_original_order(phi_total_tree)

            # 5. Apply self-integrals (diagonal correction)
            if self_integrals is not None:
                V = V + self_integrals * sigma * areas

        log_fmm_event(
            self.logger,
            "fmm_bem_apply_end",
            N=N,
            backend="tier3_fmm_scaffold",
        )
        return V.to(device=device)


def create_bem_fmm_backend(
    cfg: Optional[FmmConfig] = None, logger: Optional[object] = None
) -> FmmBemBackend:
    """Factory function used by higher-level code to get a BEM FMM backend.

    This is the function that ``bem_kernel.bem_matvec_gpu`` (or similar)
    can eventually call when ``backend="external"`` and a Tier-3 FMM
    implementation is requested.
    """
    if cfg is None:
        cfg = FmmConfig()
    return FmmBemBackend(cfg=cfg, logger=logger)