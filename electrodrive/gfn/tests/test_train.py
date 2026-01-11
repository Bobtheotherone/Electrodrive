"""Training smoke tests for GFlowNet."""

from __future__ import annotations

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv, SpecMetadata
from electrodrive.gfn.policy import (
    LogZNet,
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
)
from electrodrive.gfn.replay import MAPElitesArchive, PrefixReplay, TrajectoryReplay
from electrodrive.gfn.reward import RewardComputer, RewardConfig, RewardNormalizer
from electrodrive.gfn.rollout import SpecBatchItem
from electrodrive.gfn.train import TrainConfig, train_gfn_step
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip, get_default_device


def _plane_point_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0}],
            "charges": [{"type": "point", "pos": [0.0, 0.0, 0.5], "q": 1.0}],
        }
    )


def test_train_step_updates_and_replay_sanitized() -> None:
    ensure_cuda_available_or_skip("train step CUDA test")
    device = get_default_device()
    spec = _plane_point_spec()
    spec_meta = SpecMetadata(geom_type="plane", n_dielectrics=1, bc_type="dirichlet")

    encoder = SimpleGeoEncoder(latent_dim=8, hidden_dim=16)
    spec_embedding, _, _ = encoder.encode(spec, device=device, dtype=torch.float32)
    spec_batch = [
        SpecBatchItem(spec=spec, spec_meta=spec_meta, spec_embedding=spec_embedding, seed=11),
        SpecBatchItem(spec=spec, spec_meta=spec_meta, spec_embedding=spec_embedding, seed=13),
    ]

    env = ElectrodriveProgramEnv(device=device, max_length=3, min_length_for_stop=1)
    table = build_action_factor_table(env, device=device)
    sizes = action_factor_sizes_from_table(table)
    policy_cfg = PolicyNetConfig(spec_dim=spec_embedding.numel(), max_seq_len=env.max_length)
    policy = PolicyNet(policy_cfg, sizes, device=device, token_vocab_size=env.token_vocab_size)
    logz = LogZNet(spec_dim=spec_embedding.numel(), device=device)

    optimizer = torch.optim.Adam(list(policy.parameters()) + list(logz.parameters()), lr=1e-3)

    reward_config = RewardConfig(n_points=16, ratio_boundary=0.5, max_iter=20, latency_warmup=0)
    reward_computer = RewardComputer(device=device, config=reward_config)
    reward_normalizer = RewardNormalizer(device=device)

    trajectory_replay = TrajectoryReplay(capacity=16, seed=0)
    prefix_replay = PrefixReplay(capacity=16, seed=0)
    archive = MAPElitesArchive(seed=0)

    train_config = TrainConfig(max_steps=3, replay_batch_size=2, replay_weight=0.5)

    before = [p.detach().clone() for p in policy.parameters()]
    metrics = train_gfn_step(
        env=env,
        policy=policy,
        logz=logz,
        spec_batch=spec_batch,
        reward_computer=reward_computer,
        optimizer=optimizer,
        trajectory_replay=trajectory_replay,
        prefix_replay=prefix_replay,
        archive=archive,
        config=train_config,
        reward_normalizer=reward_normalizer,
    )
    delta = torch.stack([(a - b).abs().sum() for a, b in zip(policy.parameters(), before)]).sum()
    assert torch.isfinite(metrics["loss"]).item()
    assert delta.item() > 0.0
    assert len(trajectory_replay) > 0

    metrics_replay = train_gfn_step(
        env=env,
        policy=policy,
        logz=logz,
        spec_batch=spec_batch,
        reward_computer=reward_computer,
        optimizer=optimizer,
        trajectory_replay=trajectory_replay,
        prefix_replay=prefix_replay,
        archive=archive,
        config=train_config,
        reward_normalizer=reward_normalizer,
    )
    assert metrics_replay["replay_batch"].item() >= 1

    replay_item = trajectory_replay.sample(1)[0]
    assert replay_item.spec_embedding.device.type == "cpu"
    assert replay_item.action_sequence.device.type == "cpu"
    assert replay_item.logpf.device.type == "cpu"
    assert replay_item.logpb.device.type == "cpu"

    prefix_item = prefix_replay.sample(1)[0]
    assert prefix_item.state.mask_cuda is None
    assert prefix_item.state.mask_cache_key is None
    if prefix_item.state.ast_token_ids is not None:
        assert prefix_item.state.ast_token_ids.device.type == "cpu"
