"""Tests for MAP-Elites archive behavior."""

from __future__ import annotations

import torch

from electrodrive.gfn.dsl.nodes import AddPrimitiveBlock
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.env import SpecMetadata
from electrodrive.gfn.replay import MAPElitesArchive
from electrodrive.gfn.reward import RewardTerms


def _reward(logR: float) -> RewardTerms:
    return RewardTerms(
        relerr=torch.tensor(0.0),
        latency_ms=torch.tensor(0.0),
        instability=torch.tensor(0.0),
        complexity=torch.tensor(0.0),
        novelty=torch.tensor(0.1),
        logR=torch.tensor(logR),
    )


def test_archive_inserts_and_keeps_topk() -> None:
    archive = MAPElitesArchive(max_per_cell=2, families=("baseline",))
    program = Program(nodes=(AddPrimitiveBlock(family_name="baseline", conductor_id=0, motif_id=0),))
    spec_meta = SpecMetadata(geom_type="plane", n_dielectrics=1, bc_type="dirichlet")
    action_seq = torch.zeros((1, 8), dtype=torch.int32)
    program_hash = program.hash("spec")

    key = archive.insert(
        program=program,
        program_hash=program_hash,
        action_sequence=action_seq,
        reward_terms=_reward(0.5),
        spec_meta=spec_meta,
    )
    archive.insert(
        program=program,
        program_hash=program_hash + "b",
        action_sequence=action_seq,
        reward_terms=_reward(1.5),
        spec_meta=spec_meta,
    )
    archive.insert(
        program=program,
        program_hash=program_hash + "c",
        action_sequence=action_seq,
        reward_terms=_reward(-1.0),
        spec_meta=spec_meta,
    )

    cell = archive.get_cell(key)
    assert len(cell) == 2
    assert cell[0].score >= cell[1].score
