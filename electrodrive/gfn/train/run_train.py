"""Minimal GFlowNet training entrypoint for rich grammar experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import random

import torch

from electrodrive.flows.schemas import SCHEMA_COMPLEX_DEPTH
from electrodrive.gfn.dsl import Grammar
from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.integration.gfn_basis_generator import _spec_metadata_from_spec
from electrodrive.gfn.policy import (
    LogZNet,
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
)
from electrodrive.gfn.replay import MAPElitesArchive, PrefixReplay, TrajectoryReplay
from electrodrive.gfn.reward.gate_proxy_reward import (
    GateProxyRewardComputer,
    GateProxyRewardConfig,
    GateProxyRewardWeights,
)
from electrodrive.gfn.reward.reward import RewardTerms
from electrodrive.gfn.rollout import SpecBatchItem
from electrodrive.gfn.train import TrainConfig, train_gfn_step
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import get_default_device


class ConstantRewardComputer:
    """Deterministic reward stub for smoke training."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.param_sampler = None
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_diagnostics_batch: list[Dict[str, Any]] = []

    def compute(  # type: ignore[no-untyped-def]
        self,
        program,
        spec,
        device: Optional[torch.device] = None,
        *,
        seed: Optional[int] = None,
        spec_embedding: Optional[torch.Tensor] = None,
        param_payload: Optional[object] = None,
    ) -> RewardTerms:
        _ = spec, seed, spec_embedding, param_payload
        dev = device or self.device
        length = float(len(getattr(program, "nodes", []) or []))
        logR = torch.tensor(length, device=dev)
        zeros = torch.zeros((), device=dev)
        return RewardTerms(
            relerr=zeros,
            latency_ms=zeros,
            instability=zeros,
            complexity=zeros,
            novelty=zeros,
            logR=logR,
        )


def _default_layered_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1, -1, -2], [1, 1, 2]]},
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.5, "z_max": 2.0},
                {"name": "slab", "epsilon": 4.0, "z_min": 0.0, "z_max": 0.5},
                {"name": "region3", "epsilon": 1.0, "z_min": -2.0, "z_max": 0.0},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )


def _coerce_tuple(value: Optional[object]) -> Optional[tuple]:
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _build_grammar(cfg: Mapping[str, Any]) -> Grammar:
    base = Grammar()
    kwargs: Dict[str, Any] = {}
    for key in ("families", "motifs", "approx_types"):
        if key in cfg:
            kwargs[key] = tuple(cfg[key])
    for key in (
        "schema_ids",
        "interface_id_choices",
        "pole_count_choices",
        "branch_budget_choices",
        "conjugate_ref_choices",
    ):
        if key in cfg:
            kwargs[key] = tuple(int(v) for v in cfg[key])
    for key in ("base_pole_budget", "branch_cut_budget"):
        if key in cfg:
            kwargs[key] = int(cfg[key])
    if "primitive_schema_ids" in cfg:
        kwargs["primitive_schema_ids"] = _coerce_tuple(cfg["primitive_schema_ids"])
    if "dcim_schema_ids" in cfg:
        kwargs["dcim_schema_ids"] = _coerce_tuple(cfg["dcim_schema_ids"])
    if not kwargs:
        return base
    return Grammar(**kwargs)


def _parse_dtype(value: object) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("fp32", "float32"):
            return torch.float32
        if key in ("fp16", "float16"):
            return torch.float16
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp64", "float64", "double"):
            return torch.float64
    return torch.float32


def _build_gate_proxy_config(cfg: Mapping[str, Any]) -> GateProxyRewardConfig:
    weights_cfg = cfg.get("weights", {}) if isinstance(cfg.get("weights", {}), Mapping) else {}
    weights = GateProxyRewardWeights(
        gateA=float(weights_cfg.get("gateA", GateProxyRewardWeights().gateA)),
        gateB=float(weights_cfg.get("gateB", GateProxyRewardWeights().gateB)),
        gateC=float(weights_cfg.get("gateC", GateProxyRewardWeights().gateC)),
        gateD=float(weights_cfg.get("gateD", GateProxyRewardWeights().gateD)),
        speed=float(weights_cfg.get("speed", GateProxyRewardWeights().speed)),
        complexity=float(weights_cfg.get("complexity", GateProxyRewardWeights().complexity)),
        dcim_bonus=float(weights_cfg.get("dcim_bonus", GateProxyRewardWeights().dcim_bonus)),
        complex_bonus=float(weights_cfg.get("complex_bonus", GateProxyRewardWeights().complex_bonus)),
    )
    kwargs: Dict[str, Any] = {"weights": weights}
    for key in (
        "n_points",
        "ratio_boundary",
        "supervision_mode",
        "reg_l2",
        "logR_clip",
        "nonfinite_penalty",
        "max_term",
        "collocation_cache_size",
        "gateA_n_interior",
        "gateA_exclusion_radius",
        "gateA_interface_band",
        "gateA_fd_h",
        "gateA_prefer_autograd",
        "gateA_autograd_max_samples",
        "gateA_fd_max_samples",
        "gateA_transform",
        "gateA_cap",
        "gateA_linf_tol",
        "gateB_n_xy",
        "interface_delta",
        "gateC_n_dir",
        "near_radii",
        "far_radii",
        "gateD_n_points",
        "stability_delta",
        "use_reference_potential",
        "param_fallback",
        "fallback_latent_dim",
        "fallback_schema_id",
        "dcim_term_cost",
        "dcim_block_cost",
    ):
        if key in cfg:
            kwargs[key] = cfg[key]
    if "dtype" in cfg:
        kwargs["dtype"] = _parse_dtype(cfg["dtype"])
    if "near_radii" in kwargs:
        kwargs["near_radii"] = tuple(float(v) for v in kwargs["near_radii"])
    if "far_radii" in kwargs:
        kwargs["far_radii"] = tuple(float(v) for v in kwargs["far_radii"])
    if "logR_clip" in kwargs:
        kwargs["logR_clip"] = tuple(float(v) for v in kwargs["logR_clip"])
    return GateProxyRewardConfig(**kwargs)


def run_train_from_config(config: Mapping[str, Any]) -> Path:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_default_device()
    spec_dim = int(config.get("spec_dim", 8))
    steps = int(config.get("steps", 2))
    batch_size = int(config.get("batch_size", 4))
    lr = float(config.get("lr", 1e-3))

    grammar_cfg = config.get("grammar", {}) or {}
    grammar = _build_grammar(grammar_cfg)

    max_length = int(config.get("max_length", 6))
    min_length_for_stop = int(config.get("min_length_for_stop", 3))
    env = ElectrodriveProgramEnv(
        grammar=grammar,
        max_length=max_length,
        min_length_for_stop=min_length_for_stop,
        device=device,
    )

    factor_table = build_action_factor_table(env, device=device)
    factor_sizes = action_factor_sizes_from_table(factor_table)
    policy_cfg = PolicyNetConfig(
        spec_dim=spec_dim,
        max_seq_len=max_length,
        token_vocab_size=env.token_vocab_size,
    )
    policy = PolicyNet(policy_cfg, factor_sizes, device=device, token_vocab_size=env.token_vocab_size)
    logz = LogZNet(spec_dim=spec_dim, device=device)

    optimizer = torch.optim.Adam(list(policy.parameters()) + list(logz.parameters()), lr=lr)

    spec_payload = config.get("spec")
    spec = CanonicalSpec.from_json(spec_payload) if spec_payload else _default_layered_spec()
    extra_overrides = dict(config.get("spec_extra", {}) or {})
    if "allow_real_primitives" not in extra_overrides:
        extra_overrides["allow_real_primitives"] = False
    spec_meta = _spec_metadata_from_spec(spec, extra_overrides=extra_overrides)

    encoder = SimpleGeoEncoder(latent_dim=spec_dim, hidden_dim=max(16, spec_dim))
    spec_embedding, _, _ = encoder.encode(spec, device=device, dtype=torch.float32)
    spec_batch = [
        SpecBatchItem(
            spec=spec,
            spec_meta=spec_meta,
            spec_embedding=spec_embedding,
            seed=seed + idx,
        )
        for idx in range(batch_size)
    ]

    reward_cfg = config.get("reward", {}) or {}
    reward_type = str(reward_cfg.get("type", "constant")).strip().lower()
    if reward_type == "gate_proxy":
        reward_computer = GateProxyRewardComputer(
            device=device,
            config=_build_gate_proxy_config(reward_cfg),
        )
    elif reward_type == "constant":
        reward_computer = ConstantRewardComputer(device=device)
    else:
        raise ValueError(f"Unknown reward.type '{reward_type}' (expected 'constant' or 'gate_proxy').")
    trajectory_replay = TrajectoryReplay(capacity=max(32, batch_size * 4), seed=seed)
    prefix_replay = PrefixReplay(capacity=max(32, batch_size * 4), seed=seed)
    archive = MAPElitesArchive(seed=seed)
    train_cfg = TrainConfig(max_steps=max_length, replay_batch_size=0, replay_weight=0.0)

    for _ in range(max(1, steps)):
        train_gfn_step(
            env=env,
            policy=policy,
            logz=logz,
            spec_batch=spec_batch,
            reward_computer=reward_computer,
            optimizer=optimizer,
            trajectory_replay=trajectory_replay,
            prefix_replay=prefix_replay,
            archive=archive,
            config=train_cfg,
        )

    out_path = Path(config.get("output_path", "artifacts/gfn_rich_ckpt.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state": policy.state_dict(),
        "logz_state": logz.state_dict(),
        "policy_config": asdict(policy_cfg),
        "grammar": {
            "families": grammar.families,
            "motifs": grammar.motifs,
            "approx_types": grammar.approx_types,
            "schema_ids": grammar.schema_ids,
            "base_pole_budget": grammar.base_pole_budget,
            "branch_cut_budget": grammar.branch_cut_budget,
            "interface_id_choices": grammar.interface_id_choices,
            "pole_count_choices": grammar.pole_count_choices,
            "branch_budget_choices": grammar.branch_budget_choices,
            "primitive_schema_ids": grammar.primitive_schema_ids,
            "dcim_schema_ids": grammar.dcim_schema_ids,
            "conjugate_ref_choices": grammar.conjugate_ref_choices,
            "max_length": max_length,
            "min_length_for_stop": min_length_for_stop,
        },
        "action_vocab": grammar.action_vocab(),
        "seed": seed,
        "spec_dim": spec_dim,
        "complex_schema_id": SCHEMA_COMPLEX_DEPTH,
    }
    torch.save(payload, out_path)
    return out_path


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required for train_gfn_rich config parsing.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal rich GFN checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config for GFN training.")
    args = parser.parse_args()
    config = _load_yaml(Path(args.config))
    out_path = run_train_from_config(config)
    print(f"Saved GFN checkpoint to {out_path}")


if __name__ == "__main__":
    main()
