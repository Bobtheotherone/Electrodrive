"""Shared rollout data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from electrodrive.gfn.env import PartialProgramState

@dataclass(frozen=True)
class TrajectoryBatch:
    """Batched trajectories produced by the rollout engine."""

    actions: torch.Tensor
    logpf: torch.Tensor
    logpb: torch.Tensor
    done: torch.Tensor
    lengths: torch.Tensor
    program_hashes: Tuple[str, ...]
    state_hashes: Tuple[Tuple[str, ...], ...]
    final_states: Optional[Tuple[PartialProgramState, ...]] = None

    def to(self, device: torch.device) -> "TrajectoryBatch":
        """Move tensor fields to the specified device."""
        return TrajectoryBatch(
            actions=self.actions.to(device),
            logpf=self.logpf.to(device),
            logpb=self.logpb.to(device),
            done=self.done.to(device),
            lengths=self.lengths.to(device),
            program_hashes=self.program_hashes,
            state_hashes=self.state_hashes,
            final_states=self.final_states,
        )


__all__ = ["TrajectoryBatch"]
