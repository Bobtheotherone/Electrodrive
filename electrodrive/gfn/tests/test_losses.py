"""Tests for GFlowNet losses."""

from __future__ import annotations

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.losses import tb_loss
from electrodrive.gfn.rollout.types import TrajectoryBatch
from electrodrive.utils.device import ensure_cuda_available_or_skip, get_default_device


def test_tb_loss_is_finite() -> None:
    ensure_cuda_available_or_skip("tb loss CUDA test")
    device = get_default_device()
    env = ElectrodriveProgramEnv(device=device)
    spec_meta = SpecMetadata(geom_type="plane", n_dielectrics=1, bc_type="dirichlet")
    states = [env.reset("spec", spec_meta) for _ in range(3)]

    batch = 3
    steps = 4
    logpf = torch.randn((batch, steps), device=device)
    logpb = torch.randn((batch, steps), device=device)
    actions = torch.zeros((batch, steps, env.ACTION_ENCODING_SIZE), dtype=torch.int32, device=device)
    done = torch.zeros((batch, steps), dtype=torch.bool, device=device)
    lengths = torch.full((batch,), steps, dtype=torch.long, device=device)
    program_hashes = tuple(state.program.hash(state.spec_hash) for state in states)
    state_hashes = tuple(tuple(state.state_hash for _ in range(steps)) for state in states)
    trajectories = TrajectoryBatch(
        actions=actions,
        logpf=logpf,
        logpb=logpb,
        done=done,
        lengths=lengths,
        program_hashes=program_hashes,
        state_hashes=state_hashes,
        final_states=tuple(states),
    )
    logZ = torch.randn((batch,), device=device)
    logR = torch.randn((batch,), device=device)
    loss = tb_loss(trajectories, logZ, logR)
    assert torch.isfinite(loss).item()
