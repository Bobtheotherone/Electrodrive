"""Multi-GPU runtime helpers for FMM.

Target size: ~1600 LOC.

Responsibilities
----------------
- Discover and manage multiple GPUs on a single node.
- Coordinate:
    * device selection
    * stream and event management
    * peer-to-peer transfers
- Integrate with MPI domain decomposition (one or more GPUs per rank).
- Provide a thin abstraction that higher-level FMM components can
  use without hard-coding CUDA device IDs.

For now the implementation is intentionally minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int


def list_gpus() -> List[GpuInfo]:
    """Return a list of detected GPUs using PyTorch APIs."""
    if not torch.cuda.is_available():
        return []
    infos: List[GpuInfo] = []
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        infos.append(
            GpuInfo(
                index=idx,
                name=str(prop.name),
                total_memory_bytes=int(prop.total_memory),
            )
        )
    return infos
