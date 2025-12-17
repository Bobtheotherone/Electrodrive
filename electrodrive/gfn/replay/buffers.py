"""Replay buffers for GFlowNet trajectories and prefixes."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence

import torch

from electrodrive.gfn.env import PartialProgramState
from electrodrive.gfn.reward.reward import RewardTerms


@dataclass(frozen=True)
class TrajectoryReplayItem:
    """CPU-safe trajectory snapshot for replay."""

    program_hash: str
    spec_hash: str
    spec_embedding: torch.Tensor
    action_sequence: torch.Tensor
    logpf: torch.Tensor
    logpb: torch.Tensor
    reward_terms: RewardTerms
    length: int


@dataclass(frozen=True)
class PrefixReplayItem:
    """CPU-safe prefix snapshot for replay."""

    state: PartialProgramState
    action_prefix: torch.Tensor
    state_hash: str
    program_hash: str


def sanitize_state_for_replay(state: PartialProgramState) -> PartialProgramState:
    """Strip CUDA caches and move tokens to CPU for replay storage."""
    token_ids = state.ast_token_ids
    if token_ids is not None and token_ids.device.type != "cpu":
        token_ids = token_ids.detach().to("cpu")
    return PartialProgramState(
        spec_hash=state.spec_hash,
        spec_meta=state.spec_meta,
        ast_partial=state.ast_partial,
        action_mask=state.action_mask,
        mask_cuda=None,
        mask_cache_key=None,
        ast_token_ids=token_ids,
        cached_embeddings=None,
    )


class TrajectoryReplay:
    """Trajectory replay buffer with deduplication."""

    def __init__(self, capacity: int = 10000, *, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._items: List[TrajectoryReplayItem] = []
        self._dedupe: set[str] = set()
        self._priorities: List[float] = []

    def add(self, item: TrajectoryReplayItem) -> bool:
        """Add a trajectory if its program hash is not already stored."""
        if item.program_hash in self._dedupe:
            return False
        if len(self._items) >= self.capacity:
            evicted = self._items.pop(0)
            self._priorities.pop(0)
            self._dedupe.discard(evicted.program_hash)
        spec_embedding = item.spec_embedding
        if spec_embedding.device.type != "cpu":
            spec_embedding = spec_embedding.detach().to("cpu")
        action_sequence = item.action_sequence
        if action_sequence.device.type != "cpu":
            action_sequence = action_sequence.detach().to("cpu")
        logpf = item.logpf
        if logpf.device.type != "cpu":
            logpf = logpf.detach().to("cpu")
        logpb = item.logpb
        if logpb.device.type != "cpu":
            logpb = logpb.detach().to("cpu")
        reward_terms = item.reward_terms.detach_cpu()
        self._items.append(
            TrajectoryReplayItem(
                program_hash=item.program_hash,
                spec_hash=item.spec_hash,
                spec_embedding=spec_embedding,
                action_sequence=action_sequence,
                logpf=logpf,
                logpb=logpb,
                reward_terms=reward_terms,
                length=item.length,
            )
        )
        self._dedupe.add(item.program_hash)
        self._priorities.append(float(reward_terms.logR))
        return True

    def sample(self, batch_size: int, *, prioritized: bool = False) -> List[TrajectoryReplayItem]:
        """Sample a batch of trajectories."""
        if not self._items or batch_size <= 0:
            return []
        if not prioritized:
            return self._rng.sample(self._items, k=min(batch_size, len(self._items)))
        weights = torch.tensor(self._priorities, dtype=torch.float)
        if weights.numel() == 0:
            return []
        weights = weights - weights.max()
        probs = torch.softmax(weights, dim=0)
        replacement = batch_size > len(self._items)
        indices = torch.multinomial(probs, batch_size, replacement=replacement)
        return [self._items[int(idx)] for idx in indices]

    def __len__(self) -> int:
        return len(self._items)


class PrefixReplay:
    """Replay buffer for sanitized prefix states."""

    def __init__(self, capacity: int = 50000, *, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._items: List[PrefixReplayItem] = []
        self._dedupe: set[str] = set()

    def add(self, state: PartialProgramState, action_prefix: torch.Tensor, program_hash: str) -> bool:
        """Add a prefix state after sanitization."""
        sanitized = sanitize_state_for_replay(state)
        state_hash = sanitized.state_hash
        if state_hash in self._dedupe:
            return False
        if len(self._items) >= self.capacity:
            evicted = self._items.pop(0)
            self._dedupe.discard(evicted.state_hash)
        if action_prefix.device.type != "cpu":
            action_prefix = action_prefix.detach().to("cpu")
        self._items.append(
            PrefixReplayItem(
                state=sanitized,
                action_prefix=action_prefix,
                state_hash=state_hash,
                program_hash=program_hash,
            )
        )
        self._dedupe.add(state_hash)
        return True

    def sample(self, batch_size: int) -> List[PrefixReplayItem]:
        """Sample a batch of prefixes."""
        if not self._items or batch_size <= 0:
            return []
        return self._rng.sample(self._items, k=min(batch_size, len(self._items)))

    def __len__(self) -> int:
        return len(self._items)


__all__ = [
    "TrajectoryReplayItem",
    "PrefixReplayItem",
    "sanitize_state_for_replay",
    "TrajectoryReplay",
    "PrefixReplay",
]
