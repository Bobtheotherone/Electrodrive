"""Quality-diversity archive for GFlowNet program candidates."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.env import SpecMetadata
from electrodrive.gfn.reward.reward import RewardTerms


@dataclass(frozen=True)
class ArchiveKey:
    """MAP-Elites archive key."""

    length_bucket: int
    family_signature: Tuple[Tuple[str, int], ...]
    novelty_bucket: int
    conditioning_bucket: int


@dataclass(frozen=True)
class ArchiveEntry:
    """CPU-safe archive entry."""

    program_hash: str
    action_sequence: torch.Tensor
    reward_terms: RewardTerms
    length: int
    score: float


class MAPElitesArchive:
    """MAP-Elites style archive keyed by program and conditioning signatures."""

    def __init__(
        self,
        *,
        max_per_cell: int = 5,
        length_bin: int = 4,
        novelty_bins: int = 10,
        families: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.max_per_cell = max_per_cell
        self.length_bin = length_bin
        self.novelty_bins = novelty_bins
        self.families = tuple(families) if families is not None else ()
        self._cells: Dict[ArchiveKey, List[ArchiveEntry]] = {}
        self._rng = random.Random(seed)

    def insert(
        self,
        *,
        program: Program,
        program_hash: str,
        action_sequence: torch.Tensor,
        reward_terms: RewardTerms,
        spec_meta: SpecMetadata,
    ) -> ArchiveKey:
        """Insert an entry into the archive, keeping top-k per cell."""
        key = self._make_key(program, reward_terms, spec_meta)
        if action_sequence.device.type != "cpu":
            action_sequence = action_sequence.detach().to("cpu")
        entry = ArchiveEntry(
            program_hash=program_hash,
            action_sequence=action_sequence,
            reward_terms=reward_terms.detach_cpu(),
            length=len(program),
            score=float(reward_terms.logR),
        )
        cell = self._cells.setdefault(key, [])
        cell.append(entry)
        cell.sort(key=lambda e: e.score, reverse=True)
        if len(cell) > self.max_per_cell:
            del cell[self.max_per_cell :]
        return key

    def get_cell(self, key: ArchiveKey) -> List[ArchiveEntry]:
        """Return the archive entries for a cell."""
        return list(self._cells.get(key, []))

    def sample(self, batch_size: int) -> List[ArchiveEntry]:
        """Sample random entries across all cells."""
        entries = [entry for cell in self._cells.values() for entry in cell]
        if not entries or batch_size <= 0:
            return []
        return self._rng.sample(entries, k=min(batch_size, len(entries)))

    def __len__(self) -> int:
        return sum(len(cell) for cell in self._cells.values())

    def _make_key(self, program: Program, reward_terms: RewardTerms, spec_meta: SpecMetadata) -> ArchiveKey:
        length_bucket = len(program) // max(self.length_bin, 1)
        family_signature = self._family_signature(program)
        novelty = float(reward_terms.novelty)
        novelty_bucket = int(max(0.0, min(self.novelty_bins - 1, novelty * self.novelty_bins)))
        conditioning_bucket = int(spec_meta.n_dielectrics)
        return ArchiveKey(
            length_bucket=length_bucket,
            family_signature=family_signature,
            novelty_bucket=novelty_bucket,
            conditioning_bucket=conditioning_bucket,
        )

    def _family_signature(self, program: Program) -> Tuple[Tuple[str, int], ...]:
        counts: Dict[str, int] = {name: 0 for name in self.families}
        for node in program.nodes:
            if isinstance(node, AddPrimitiveBlock):
                counts[node.family_name] = counts.get(node.family_name, 0) + 1
        if self.families:
            return tuple((name, counts.get(name, 0)) for name in self.families)
        return tuple(sorted(counts.items()))


__all__ = ["ArchiveKey", "ArchiveEntry", "MAPElitesArchive"]
