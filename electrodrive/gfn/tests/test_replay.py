"""Tests for replay buffers and sanitization."""

from __future__ import annotations

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.replay import TrajectoryReplay, TrajectoryReplayItem, sanitize_state_for_replay
from electrodrive.gfn.reward import RewardTerms
from electrodrive.utils.device import ensure_cuda_available_or_skip, get_default_device


def test_trajectory_replay_dedupes() -> None:
    replay = TrajectoryReplay(capacity=4)
    reward_terms = RewardTerms(
        relerr=torch.tensor(0.0),
        latency_ms=torch.tensor(0.0),
        instability=torch.tensor(0.0),
        complexity=torch.tensor(0.0),
        novelty=torch.tensor(0.0),
        logR=torch.tensor(0.0),
    )
    item = TrajectoryReplayItem(
        program_hash="hash",
        spec_hash="spec",
        spec_embedding=torch.zeros((4,)),
        action_sequence=torch.zeros((1, 8), dtype=torch.int32),
        logpf=torch.zeros((1,)),
        logpb=torch.zeros((1,)),
        reward_terms=reward_terms,
        length=1,
    )
    assert replay.add(item)
    assert not replay.add(item)
    assert len(replay) == 1


def test_replay_sanitization_clears_cuda_cache() -> None:
    ensure_cuda_available_or_skip("replay sanitization CUDA test")
    device = get_default_device()
    env = ElectrodriveProgramEnv(device=device)
    spec_meta = SpecMetadata(geom_type="plane", n_dielectrics=1, bc_type="dirichlet")
    state = env.reset("spec", spec_meta)
    _ = env.get_action_mask(state)
    assert state.mask_cuda is not None
    sanitized = sanitize_state_for_replay(state)
    assert sanitized.mask_cuda is None
    assert sanitized.mask_cache_key is None
    if sanitized.ast_token_ids is not None:
        assert sanitized.ast_token_ids.device.type == "cpu"
