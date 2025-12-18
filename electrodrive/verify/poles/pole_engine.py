from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch

from .backend_denominator_roots import find_poles_denominator_roots
from .backend_rational_fit import find_poles_rational_fit
from .pole_types import PoleTerm


@dataclass(frozen=True)
class PoleSearchConfig:
    method: str = "denominator_roots"
    max_poles: int = 4
    k_rect: float = 20.0
    n_samples: int = 256
    newton_tol: float = 1e-8
    newton_max_iter: int = 12
    rational_order: Optional[int] = None
    stability_half_plane: Optional[str] = None  # e.g., "Re>0" or "Im>0"


def find_poles(
    reflection_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    denominator_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
    k_samples: Optional[Iterable[float]] = None,
    config: Optional[PoleSearchConfig] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex128,
) -> List[PoleTerm]:
    cfg = config or PoleSearchConfig()
    method = cfg.method.lower()
    if method == "denominator_roots":
        return find_poles_denominator_roots(
            denominator_fn or reflection_fn,  # type: ignore[arg-type]
            k_samples=k_samples,
            max_poles=cfg.max_poles,
            k_rect=cfg.k_rect,
            n_samples=cfg.n_samples,
            newton_tol=cfg.newton_tol,
            newton_max_iter=cfg.newton_max_iter,
            device=device,
            dtype=dtype,
        )
    if method == "rational_fit":
        return find_poles_rational_fit(
            reflection_fn,
            k_samples=k_samples,
            max_poles=cfg.max_poles,
            order_override=cfg.rational_order,
            stability_half_plane=cfg.stability_half_plane,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown pole search method '{cfg.method}'")
