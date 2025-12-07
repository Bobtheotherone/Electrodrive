"""Benchmark harness for the FMM subsystem.

Target size: ~1200 LOC.

Responsibilities
----------------
- Provide callable functions / CLIs to:
    * benchmark matvec throughput vs N
    * measure scaling with number of GPUs / MPI ranks
    * compare CPU vs GPU vs KeOps vs direct BEM
- Report:
    * wall times
    * iteration counts
    * accuracy vs direct O(N^2) reference.

This module is a key part of making the Tier-3 FMM *observable* and
driving performance engineering in a disciplined way.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

import torch
from torch import Tensor

from .config import FmmConfig
from .bem_coupling import FmmBemBackend


def benchmark_single_run(
    backend_factory: Callable[[FmmConfig], FmmBemBackend],
    cfg: FmmConfig,
    N: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run a single placeholder benchmark.

    The full implementation will:
    - generate random points / charges
    - construct the tree and interaction lists
    - run one or more matvecs
    - optionally compare against a direct O(N^2) result.
    """
    t0 = time.perf_counter()
    cfg.validate()
    backend = backend_factory(cfg)

    # Placeholder: allocate dummy data and call a no-op matvec.
    points = torch.zeros(N, 3, device=device)
    charges = torch.zeros(N, device=device)
    _ = backend.apply(points, charges)

    t1 = time.perf_counter()
    return {"wall_time_s": t1 - t0}
