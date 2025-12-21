"""Training loop skeleton for GFlowNet policy optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.losses import db_loss, subtb_loss, tb_loss
from electrodrive.gfn.policy import LogZNet, PolicyNet
from electrodrive.gfn.replay import MAPElitesArchive, PrefixReplay, TrajectoryReplay, TrajectoryReplayItem
from electrodrive.gfn.reward import RewardComputer, RewardNormalizer
from electrodrive.gfn.rollout import SpecBatchItem, TemperatureSchedule, TrajectoryBatch, rollout_on_policy


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a single GFlowNet training step."""

    max_steps: int = 32
    tb_weight: float = 1.0
    db_weight: float = 0.0
    subtb_weight: float = 0.0
    replay_batch_size: int = 16
    replay_weight: float = 0.5
    replay_prioritized: bool = True
    reward_clip: Optional[tuple[float, float]] = (-20.0, 20.0)
    temperature_schedule: TemperatureSchedule = field(default_factory=TemperatureSchedule)


def train_gfn_step(
    *,
    env: ElectrodriveProgramEnv,
    policy: PolicyNet,
    logz: LogZNet,
    spec_batch: Sequence[SpecBatchItem],
    reward_computer: RewardComputer,
    optimizer: torch.optim.Optimizer,
    trajectory_replay: TrajectoryReplay,
    prefix_replay: PrefixReplay,
    archive: MAPElitesArchive,
    config: TrainConfig,
    reward_normalizer: Optional[RewardNormalizer] = None,
) -> Mapping[str, torch.Tensor]:
    """Run a single on-policy training step and update replay buffers."""
    device = policy.device
    replay_items = (
        trajectory_replay.sample(config.replay_batch_size, prioritized=config.replay_prioritized)
        if config.replay_batch_size > 0
        else []
    )
    replay_batch = _build_replay_batch(replay_items, device, env.ACTION_ENCODING_SIZE)
    trajectories = rollout_on_policy(
        env,
        policy,
        spec_batch,
        max_steps=config.max_steps,
        temperature_schedule=config.temperature_schedule,
    )
    if trajectories.final_states is None:
        raise ValueError("Rollout did not return final states for reward computation.")
    spec_embeddings = torch.stack([item.spec_embedding.detach().to(device) for item in spec_batch], dim=0)
    logZ = _forward_logz(logz, spec_embeddings, device)

    reward_terms_list = []
    diag_counts = {"empty_compilation": 0, "solver_failed": 0, "nonfinite": 0}
    if reward_computer.param_sampler is not None and hasattr(reward_computer, "compute_batch"):
        programs = []
        specs = []
        spec_embeddings_list = []
        seeds = []
        for state, spec_item in zip(trajectories.final_states, spec_batch):
            if state is None:
                raise ValueError("Rollout produced empty final states.")
            programs.append(state.program)
            specs.append(spec_item.spec)
            spec_embeddings_list.append(spec_item.spec_embedding)
            seeds.append(spec_item.seed)
        reward_terms_list = reward_computer.compute_batch(
            programs,
            specs,
            device=device,
            seeds=seeds,
            spec_embeddings=spec_embeddings_list,
        )
        for last_diag in getattr(reward_computer, "last_diagnostics_batch", []):
            for key in diag_counts:
                if last_diag.get(key):
                    diag_counts[key] += 1
    else:
        for state, spec_item in zip(trajectories.final_states, spec_batch):
            if state is None:
                raise ValueError("Rollout produced empty final states.")
            reward_terms_list.append(
                reward_computer.compute(
                    state.program,
                    spec_item.spec,
                    device=device,
                    seed=spec_item.seed,
                    spec_embedding=spec_item.spec_embedding,
                )
            )
            last_diag = reward_computer.last_diagnostics
            for key in diag_counts:
                if last_diag.get(key):
                    diag_counts[key] += 1
    logR = torch.stack([terms.logR for terms in reward_terms_list], dim=0)

    loss_tb = tb_loss(
        trajectories,
        logZ,
        logR,
        reward_clip=config.reward_clip,
        reward_normalizer=reward_normalizer,
    )
    loss_db = (
        db_loss(trajectories) if config.db_weight > 0.0 else torch.zeros((), device=device)
    )
    loss_subtb = (
        subtb_loss(trajectories, logZ, logR, reward_clip=config.reward_clip, reward_normalizer=reward_normalizer)
        if config.subtb_weight > 0.0
        else torch.zeros((), device=device)
    )
    total_loss = (
        config.tb_weight * loss_tb
        + config.db_weight * loss_db
        + config.subtb_weight * loss_subtb
    )
    replay_loss = torch.zeros((), device=device)
    if replay_batch is not None and replay_items:
        replay_logZ = _forward_logz(
            logz,
            torch.stack([item.spec_embedding.to(device) for item in replay_items], dim=0),
            device,
        )
        replay_logR = torch.stack(
            [item.reward_terms.logR.to(device) for item in replay_items], dim=0
        )
        loss_tb_replay = tb_loss(
            replay_batch,
            replay_logZ,
            replay_logR,
            reward_clip=config.reward_clip,
            reward_normalizer=reward_normalizer,
        )
        loss_db_replay = (
            db_loss(replay_batch) if config.db_weight > 0.0 else torch.zeros((), device=device)
        )
        loss_subtb_replay = (
            subtb_loss(
                replay_batch,
                replay_logZ,
                replay_logR,
                reward_clip=config.reward_clip,
                reward_normalizer=reward_normalizer,
            )
            if config.subtb_weight > 0.0
            else torch.zeros((), device=device)
        )
        replay_loss = (
            config.tb_weight * loss_tb_replay
            + config.db_weight * loss_db_replay
            + config.subtb_weight * loss_subtb_replay
        )
        total_loss = total_loss + config.replay_weight * replay_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    lengths_cpu = trajectories.lengths.to("cpu").tolist()
    for idx, state in enumerate(trajectories.final_states):
        if state is None:
            continue
        length = int(lengths_cpu[idx])
        actions_cpu = trajectories.actions[idx, :length].detach().to("cpu")
        logpf_cpu = trajectories.logpf[idx, :length].detach().to("cpu")
        logpb_cpu = trajectories.logpb[idx, :length].detach().to("cpu")
        reward_cpu = reward_terms_list[idx].detach_cpu()
        program_hash = trajectories.program_hashes[idx]
        spec_embedding_cpu = spec_batch[idx].spec_embedding.detach().to("cpu")
        trajectory_replay.add(
            TrajectoryReplayItem(
                program_hash=program_hash,
                spec_hash=state.spec_hash,
                spec_embedding=spec_embedding_cpu,
                action_sequence=actions_cpu,
                logpf=logpf_cpu,
                logpb=logpb_cpu,
                reward_terms=reward_cpu,
                length=length,
            )
        )
        prefix_replay.add(state, actions_cpu, program_hash)
        archive.insert(
            program=state.program,
            program_hash=program_hash,
            action_sequence=actions_cpu,
            reward_terms=reward_cpu,
            spec_meta=state.spec_meta,
        )

    metrics = {
        "loss": total_loss.detach(),
        "loss_tb": loss_tb.detach(),
        "loss_db": loss_db.detach(),
        "loss_subtb": loss_subtb.detach(),
        "mean_logR": logR.mean().detach(),
        "mean_length": trajectories.lengths.float().mean().detach(),
        "replay_loss": replay_loss.detach(),
        "replay_size": torch.tensor(len(trajectory_replay), device=device),
        "replay_batch": torch.tensor(len(replay_items), device=device),
        "empty_compilation_count": torch.tensor(diag_counts["empty_compilation"], device=device),
        "solver_failed_count": torch.tensor(diag_counts["solver_failed"], device=device),
        "nonfinite_count": torch.tensor(diag_counts["nonfinite"], device=device),
    }
    return metrics


def _forward_logz(logz: LogZNet, spec_embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        return logz(spec_embeddings)


def _build_replay_batch(
    items: Sequence[TrajectoryReplayItem],
    device: torch.device,
    action_dim: int,
) -> Optional["TrajectoryBatch"]:
    if not items:
        return None
    batch_size = len(items)
    max_len = max(item.length for item in items)
    actions = torch.zeros((batch_size, max_len, action_dim), dtype=torch.int32, device=device)
    logpf = torch.zeros((batch_size, max_len), device=device)
    logpb = torch.zeros((batch_size, max_len), device=device)
    done = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
    program_hashes = []
    state_hashes = []
    for idx, item in enumerate(items):
        length = int(item.length)
        lengths[idx] = length
        if length > 0:
            actions[idx, :length] = item.action_sequence.to(device=device)
            logpf[idx, :length] = item.logpf.to(device=device)
            logpb[idx, :length] = item.logpb.to(device=device)
            done[idx, length - 1] = True
        program_hashes.append(item.program_hash)
        state_hashes.append(tuple())
    return TrajectoryBatch(
        actions=actions,
        logpf=logpf,
        logpb=logpb,
        done=done,
        lengths=lengths,
        program_hashes=tuple(program_hashes),
        state_hashes=tuple(state_hashes),
        final_states=None,
    )


__all__ = ["TrainConfig", "train_gfn_step"]
