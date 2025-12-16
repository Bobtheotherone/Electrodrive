from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from electrodrive.layers.spectral_kernels import SpectralKernelSpec
from electrodrive.layers.stack import LayerStack


@dataclass(frozen=True)
class PoleSearchConfig:
    k_rect: float = 10.0
    max_poles: int = 0
    tol: float = 1e-6
    max_iter: int = 8
    prune_threshold: float = 1e-6
    seed_strategy: str = "none"  # placeholder for future contour choices


@dataclass(frozen=True)
class PoleTerm:
    pole: complex
    residue: complex
    kind: str = "guided"
    meta: Dict[str, object] = field(default_factory=dict)


def find_poles(
    stack: LayerStack,
    kernel_spec: SpectralKernelSpec,
    cfg: PoleSearchConfig,
    device: str | torch.device = "cuda",
) -> List[PoleTerm]:
    """
    Placeholder pole finder for electrostatic layered media.

    Returns an empty list by default (static mode), but keeps the API so
    future full-wave / guided-mode extraction can be enabled via cfg.max_poles > 0.
    """
    _ = (stack, kernel_spec)  # reserved for future use
    if cfg.max_poles <= 0:
        print("find_poles: pole extraction disabled (max_poles<=0); returning empty list.")
        return []

    if cfg.seed_strategy == "placeholder":
        print("find_poles: UNSAFE PLACEHOLDER path active; no true pole search implemented.")
        device = torch.device(device)
        pole = torch.tensor(-1.0, device=device, dtype=torch.float64)
        residue = torch.tensor(0.0, device=device, dtype=torch.float64)
        return [PoleTerm(pole=complex(pole.item()), residue=complex(residue.item()), kind="placeholder")]

    raise NotImplementedError("Pole extraction backend not implemented for electrostatics; disable poles or use placeholder explicitly.")
