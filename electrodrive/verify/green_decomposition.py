from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


EvalFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class GreenDecomposition:
    """Compose singular and smooth Green's function evaluators."""

    singular: EvalFn
    smooth: Optional[EvalFn] = None

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = self.singular(x, y)
        if self.smooth is not None:
            g = g + self.smooth(x, y)
        return g

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.evaluate(x, y)
