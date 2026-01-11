"""Tests for factorized policy sampling."""

from __future__ import annotations

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.policy import (
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
    sample_actions,
)
from electrodrive.utils.device import ensure_cuda_available_or_skip, get_default_device


def test_policy_sampling_respects_masks() -> None:
    ensure_cuda_available_or_skip("policy sampling mask test")
    device = get_default_device()
    env = ElectrodriveProgramEnv(max_length=1, min_length_for_stop=1, device=device)
    table = build_action_factor_table(env, device=device)
    sizes = action_factor_sizes_from_table(table)
    policy = PolicyNet(
        PolicyNetConfig(spec_dim=8, max_seq_len=env.max_length),
        sizes,
        device=device,
        token_vocab_size=env.token_vocab_size,
    )

    spec_meta = SpecMetadata(geom_type="plane", n_dielectrics=1, bc_type="dirichlet")
    state0 = env.reset("spec0", spec_meta)
    state1, _, _ = env.step(state0, env.actions[0])

    spec_embeddings = torch.zeros((2, 8), device=device)
    sample = sample_actions(policy, env, [state0, state1], spec_embeddings, temperature=1.0)
    mask_batch, _ = env.get_action_mask_batch([state0, state1])

    assert bool(mask_batch[0, sample.action_indices[0]].item())
    assert bool(mask_batch[1, sample.action_indices[1]].item())
    assert sample.actions[1].action_type == "stop"
