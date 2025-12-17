"""GFlowNet loss functions: TB, DB, and SubTB."""

from __future__ import annotations

from typing import Optional

import torch

from electrodrive.gfn.reward.reward import RewardNormalizer, clip_log_reward
from electrodrive.gfn.rollout.types import TrajectoryBatch


def tb_loss(
    trajectories: TrajectoryBatch,
    logZ: torch.Tensor,
    logR: torch.Tensor,
    *,
    reward_clip: Optional[tuple[float, float]] = None,
    reward_normalizer: Optional[RewardNormalizer] = None,
) -> torch.Tensor:
    """Compute the trajectory balance loss."""
    device = logZ.device
    logR = _prepare_logR(logR, reward_clip, reward_normalizer).to(device)
    lengths = trajectories.lengths.to(device)
    mask = _length_mask(lengths, trajectories.logpf.shape[1], device)
    logpf_sum = (trajectories.logpf.to(device) * mask).sum(dim=1)
    logpb_sum = (trajectories.logpb.to(device) * mask).sum(dim=1)
    residual = logZ + logpf_sum - logR - logpb_sum
    return torch.mean(residual * residual)


def db_loss(
    trajectories: TrajectoryBatch,
    *,
    reward_clip: Optional[tuple[float, float]] = None,
    reward_normalizer: Optional[RewardNormalizer] = None,
) -> torch.Tensor:
    """Compute a detailed balance loss over transitions (placeholder)."""
    device = trajectories.logpf.device
    lengths = trajectories.lengths.to(device)
    mask = _length_mask(lengths, trajectories.logpf.shape[1], device)
    residual = trajectories.logpf.to(device) - trajectories.logpb.to(device)
    residual = residual * mask
    denom = mask.sum().clamp_min(1.0)
    _ = reward_clip, reward_normalizer
    return (residual * residual).sum() / denom


def subtb_loss(
    trajectories: TrajectoryBatch,
    logZ: torch.Tensor,
    logR: torch.Tensor,
    *,
    reward_clip: Optional[tuple[float, float]] = None,
    reward_normalizer: Optional[RewardNormalizer] = None,
) -> torch.Tensor:
    """Compute a sub-trajectory balance loss using terminal prefixes."""
    device = logZ.device
    logR = _prepare_logR(logR, reward_clip, reward_normalizer).to(device)
    lengths = trajectories.lengths.to(device)
    mask = _length_mask(lengths, trajectories.logpf.shape[1], device)
    logpf_cum = (trajectories.logpf.to(device) * mask).cumsum(dim=1)
    logpb_cum = (trajectories.logpb.to(device) * mask).cumsum(dim=1)
    done_mask = trajectories.done.to(device)
    if done_mask.numel() == 0:
        return torch.zeros((), device=device)
    residual = logZ.unsqueeze(1) + logpf_cum - logpb_cum - logR.unsqueeze(1)
    residual = residual * done_mask
    denom = done_mask.sum().clamp_min(1.0)
    return (residual * residual).sum() / denom


def _prepare_logR(
    logR: torch.Tensor,
    reward_clip: Optional[tuple[float, float]],
    reward_normalizer: Optional[RewardNormalizer],
) -> torch.Tensor:
    if reward_clip is not None:
        logR = clip_log_reward(logR, reward_clip[0], reward_clip[1])
    if reward_normalizer is not None:
        reward_normalizer.update(logR)
        logR = reward_normalizer.normalize(logR)
    return logR


def _length_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


__all__ = ["tb_loss", "db_loss", "subtb_loss"]
