import types
from typing import Sequence
from pathlib import Path

import torch

from electrodrive.flows.models import ParamFlowNet
from electrodrive.flows.schemas import SCHEMA_REAL_POINT
from electrodrive.flows.types import FlowConfig, ParamPayload, ProgramBatch
from electrodrive.gfn.dsl import AddPrimitiveBlock, Program
from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.env.state import PartialProgramState
from electrodrive.gfn.integration import HybridGFlowFlowGenerator
from electrodrive.gfn.integration.gfn_basis_generator import _spec_metadata_from_spec
from electrodrive.gfn.policy import (
    LogZNet,
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
)
from electrodrive.gfn.reward import RewardComputer, RewardConfig
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.images.search import discover_images
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip


def _tiny_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [{"type": "plane", "z": 0.0}],
            "charges": [{"type": "point", "pos": [0.0, 0.0, 0.5], "q": 1.0}],
        }
    )


def _write_gfn_checkpoint(path: Path, *, spec_dim: int) -> None:
    device = torch.device("cuda")
    policy_cfg = PolicyNetConfig(spec_dim=spec_dim, max_seq_len=2)
    env = ElectrodriveProgramEnv(max_length=policy_cfg.max_seq_len, min_length_for_stop=1, device=device)
    factor_table = build_action_factor_table(env, device=device)
    factor_sizes = action_factor_sizes_from_table(factor_table)
    policy = PolicyNet(policy_cfg, factor_sizes, device=device, token_vocab_size=env.token_vocab_size)
    logz = LogZNet(policy_cfg.spec_dim, device=device)
    payload = {
        "policy_state": policy.state_dict(),
        "logz_state": logz.state_dict(),
        "policy_config": {
            "spec_dim": policy_cfg.spec_dim,
            "token_embed_dim": policy_cfg.token_embed_dim,
            "hidden_dim": policy_cfg.hidden_dim,
            "dropout": policy_cfg.dropout,
            "max_seq_len": policy_cfg.max_seq_len,
            "token_vocab_size": env.token_vocab_size,
        },
        "grammar": {
            "families": env.grammar.families,
            "motifs": env.grammar.motifs,
            "approx_types": env.grammar.approx_types,
            "schema_ids": env.grammar.schema_ids,
            "base_pole_budget": env.grammar.base_pole_budget,
            "branch_cut_budget": env.grammar.branch_cut_budget,
            "interface_id_choices": env.grammar.interface_id_choices,
            "pole_count_choices": env.grammar.pole_count_choices,
            "branch_budget_choices": env.grammar.branch_budget_choices,
            "primitive_schema_ids": env.grammar.primitive_schema_ids,
            "dcim_schema_ids": env.grammar.dcim_schema_ids,
            "conjugate_ref_choices": env.grammar.conjugate_ref_choices,
            "max_length": policy_cfg.max_seq_len,
            "min_length_for_stop": 1,
        },
        "action_vocab": env.grammar.action_vocab(),
    }
    torch.save(payload, path)


def _write_flow_checkpoint(path: Path, *, spec_dim: int, latent_dim: int = 4) -> None:
    device = torch.device("cuda")
    model = ParamFlowNet(
        latent_dim=latent_dim,
        model_dim=16,
        num_schemas=8,
        ast_vocab_size=16,
        spec_embed_dim=spec_dim,
        node_feat_dim=0,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    ).to(device)
    payload = {
        "model": model.state_dict(),
        "model_config": {
            "latent_dim": latent_dim,
            "model_dim": 16,
            "num_schemas": 8,
            "ast_vocab_size": 16,
            "spec_embed_dim": spec_dim,
            "node_feat_dim": 0,
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.0,
        },
        "sampler_config": {"latent_dim": latent_dim},
    }
    torch.save(payload, path)


def test_reward_cache_key_includes_seed() -> None:
    ensure_cuda_available_or_skip("step10 reward cache key")
    device = torch.device("cuda")
    spec = _tiny_spec()
    program = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="baseline",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_REAL_POINT,
            ),
        )
    )

    class DummySampler:
        def sample(
            self,
            programs: Sequence[object] | ProgramBatch,
            spec: object,
            spec_embedding: torch.Tensor,
            *,
            seed: int | None,
            device: torch.device,
            dtype: torch.dtype,
            **cfg: object,
        ) -> ParamPayload:
            _ = spec, spec_embedding, cfg
            program_list = list(programs.programs if isinstance(programs, ProgramBatch) else programs)
            batch = len(program_list)
            u_latent = torch.full((batch, 1, 4), float(seed or 0), device=device, dtype=dtype)
            node_mask = torch.ones((batch, 1), device=device, dtype=torch.bool)
            schema_ids = torch.full((batch, 1), SCHEMA_REAL_POINT, device=device, dtype=torch.long)
            return ParamPayload(
                u_latent=u_latent,
                node_mask=node_mask,
                dim_mask=None,
                schema_ids=schema_ids,
                node_to_token=[[0] for _ in range(batch)],
                seed=seed,
                config_hash="dummy",
                device=device,
                dtype=dtype,
            )

    reward = RewardComputer(
        device=device,
        config=RewardConfig(compile_cache_size=4),
        param_sampler=DummySampler(),
        flow_config=FlowConfig(n_steps=2, solver="euler", temperature=1.0, dtype="fp32"),
        flow_checkpoint_id="dummy_flow",
    )
    spec_embedding = torch.zeros(1, device=device)

    reward._compile_program(
        program,
        spec,
        device,
        reward.config.dtype,
        reward.config,
        param_seed=11,
        spec_embedding=spec_embedding,
    )
    reward._compile_program(
        program,
        spec,
        device,
        reward.config.dtype,
        reward.config,
        param_seed=12,
        spec_embedding=spec_embedding,
    )

    assert len(reward._compile_cache) == 2


def test_hybrid_compile_smoke(tmp_path: Path) -> None:
    ensure_cuda_available_or_skip("step10 hybrid compile smoke")
    device = torch.device("cuda")
    spec = _tiny_spec()
    gfn_ckpt = tmp_path / "gfn.pt"
    flow_ckpt = tmp_path / "flow.pt"
    _write_gfn_checkpoint(gfn_ckpt, spec_dim=8)
    _write_flow_checkpoint(flow_ckpt, spec_dim=8)

    generator = HybridGFlowFlowGenerator(
        checkpoint_path=str(gfn_ckpt),
        flow_checkpoint_path=str(flow_ckpt),
        flow_config=FlowConfig(n_steps=2, solver="euler", temperature=1.0, dtype="fp32"),
        device=device,
        dtype=torch.float32,
    )

    program = Program(
        nodes=(
            AddPrimitiveBlock(
                family_name="baseline",
                conductor_id=0,
                motif_id=0,
                schema_id=SCHEMA_REAL_POINT,
            ),
        )
    )
    spec_embedding = torch.zeros(8, device=device)
    spec_meta = _spec_metadata_from_spec(spec)
    state = PartialProgramState(spec_hash="spec", spec_meta=spec_meta, ast_partial=program)
    generator._rollout = lambda *a, **k: types.SimpleNamespace(final_states=(state,))  # type: ignore[assignment]

    elems = generator.generate(spec=spec, spec_embedding=spec_embedding, n_candidates=1, seed=7)
    assert elems
    pos = elems[0].params.get("position")
    assert pos is not None and torch.isfinite(pos).all()


def test_discover_images_gfn_flow_smoke(tmp_path: Path) -> None:
    ensure_cuda_available_or_skip("step10 discover gfn_flow smoke")
    spec = _tiny_spec()
    gfn_ckpt = tmp_path / "gfn_flow_gfn.pt"
    flow_ckpt = tmp_path / "gfn_flow_flow.pt"
    _write_gfn_checkpoint(gfn_ckpt, spec_dim=32)
    _write_flow_checkpoint(flow_ckpt, spec_dim=32)

    class DummyLogger:
        def info(self, *args, **kwargs) -> None:
            pass

        def warning(self, *args, **kwargs) -> None:
            pass

        def error(self, *args, **kwargs) -> None:
            pass

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=1,
        reg_l1=1e-3,
        restarts=0,
        logger=DummyLogger(),
        n_points_override=16,
        ratio_boundary_override=0.5,
        basis_generator=None,
        basis_generator_mode="gfn_flow",
        geo_encoder=SimpleGeoEncoder(),
        gfn_checkpoint=str(gfn_ckpt),
        gfn_seed=3,
        flow_checkpoint=str(flow_ckpt),
        flow_steps=1,
        flow_solver="euler",
        flow_temp=1.0,
        flow_dtype="fp32",
        flow_seed=11,
    )

    assert hasattr(system, "elements")
