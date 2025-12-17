"""Rollout engine for batched GFlowNet trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.policy.models import PolicyNet, sample_actions
from electrodrive.gfn.rollout.types import TrajectoryBatch


@dataclass(frozen=True)
class SpecBatchItem:
    """Spec container for batched rollouts."""

    spec: object
    spec_meta: SpecMetadata
    spec_embedding: torch.Tensor
    seed: Optional[int] = None


@dataclass(frozen=True)
class TemperatureSchedule:
    """Simple linear temperature schedule."""

    start: float = 1.0
    end: float = 1.0
    decay_steps: int = 1

    def value(self, step: int) -> float:
        """Return the temperature for a given step index."""
        if self.decay_steps <= 1:
            return float(self.end)
        ratio = min(float(step) / float(self.decay_steps), 1.0)
        return float(self.start + (self.end - self.start) * ratio)


def rollout_on_policy(
    env: ElectrodriveProgramEnv,
    policy: PolicyNet,
    spec_batch: Sequence[SpecBatchItem],
    *,
    max_steps: int,
    temperature_schedule: Optional[TemperatureSchedule] = None,
    generator: Optional[torch.Generator] = None,
) -> TrajectoryBatch:
    """Generate batched rollouts using the forward policy."""
    if not spec_batch:
        raise ValueError("spec_batch must be non-empty")
    device = policy.device
    generator = generator or env.torch_gen
    states = [env.reset(item.spec, item.spec_meta, seed=item.seed) for item in spec_batch]
    batch_size = len(states)
    done_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    spec_embeddings = torch.stack([item.spec_embedding.detach().to(device) for item in spec_batch], dim=0)

    actions_steps: List[torch.Tensor] = []
    logpf_steps: List[torch.Tensor] = []
    logpb_steps: List[torch.Tensor] = []
    done_steps: List[torch.Tensor] = []
    state_hashes: List[List[str]] = [[] for _ in range(batch_size)]

    for step in range(max_steps):
        if bool(done_flags.all()):
            break
        active_idx = (~done_flags).nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            break
        active_states = [states[idx] for idx in active_idx.to("cpu").tolist()]
        temperature = temperature_schedule.value(step) if temperature_schedule else 1.0
        sample = sample_actions(
            policy,
            env,
            active_states,
            spec_embeddings.index_select(0, active_idx),
            temperature=temperature,
            generator=generator,
        )

        step_actions = torch.zeros(
            (batch_size, env.ACTION_ENCODING_SIZE), dtype=torch.int32, device=device
        )
        step_logpf = torch.zeros((batch_size,), device=device)
        step_logpb = torch.zeros((batch_size,), device=device)

        step_actions.index_copy_(0, active_idx, sample.encoded_actions)
        step_logpf.index_copy_(0, active_idx, sample.logpf)

        next_states, done_mask = env.step_batch(active_states, sample.actions)
        logpb_vals: List[torch.Tensor] = []
        for state, action, next_state in zip(active_states, sample.actions, next_states):
            logpb_vals.append(env.get_logpb(state, action, next_state))
        if logpb_vals:
            step_logpb.index_copy_(0, active_idx, torch.stack(logpb_vals).to(device))

        for idx_tensor, next_state, done in zip(active_idx, next_states, done_mask):
            idx = int(idx_tensor.item())
            states[idx] = next_state
            done_flags[idx] = bool(done)
            state_hashes[idx].append(next_state.state_hash)

        actions_steps.append(step_actions)
        logpf_steps.append(step_logpf)
        logpb_steps.append(step_logpb)
        step_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        step_done.index_copy_(0, active_idx, done_mask.to(device))
        done_steps.append(step_done)

    if not actions_steps:
        empty = torch.empty((batch_size, 0), device=device)
        return TrajectoryBatch(
            actions=torch.empty((batch_size, 0, env.ACTION_ENCODING_SIZE), device=device, dtype=torch.int32),
            logpf=empty,
            logpb=empty,
            done=empty.to(torch.bool),
            lengths=torch.zeros(batch_size, dtype=torch.long, device=device),
            program_hashes=tuple(state.program.hash(state.spec_hash) for state in states),
            state_hashes=tuple(tuple(hashes) for hashes in state_hashes),
            final_states=tuple(states),
        )

    actions = torch.stack(actions_steps, dim=1)
    logpf = torch.stack(logpf_steps, dim=1)
    logpb = torch.stack(logpb_steps, dim=1)
    done = torch.stack(done_steps, dim=1)
    lengths = torch.tensor([len(hashes) for hashes in state_hashes], dtype=torch.long, device=device)
    program_hashes = tuple(state.program.hash(state.spec_hash) for state in states)
    return TrajectoryBatch(
        actions=actions,
        logpf=logpf,
        logpb=logpb,
        done=done,
        lengths=lengths,
        program_hashes=program_hashes,
        state_hashes=tuple(tuple(hashes) for hashes in state_hashes),
        final_states=tuple(states),
    )


__all__ = ["SpecBatchItem", "TemperatureSchedule", "rollout_on_policy"]
