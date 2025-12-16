from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SpectralKernelSpec:
    source_region: int
    obs_region: int
    component: str = "potential"
    bc_kind: str = "dielectric_interfaces"


@dataclass(frozen=True)
class FitTarget:
    kind: str
    weights: str = "none"
    meta: Dict[str, object] = field(default_factory=dict)
