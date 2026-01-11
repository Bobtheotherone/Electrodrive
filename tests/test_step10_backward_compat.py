from pathlib import Path

import torch

from electrodrive.gfn.env import ElectrodriveProgramEnv
from electrodrive.gfn.integration import GFlowNetProgramGenerator
from electrodrive.gfn.policy import (
    LogZNet,
    PolicyNet,
    PolicyNetConfig,
    action_factor_sizes_from_table,
    build_action_factor_table,
)
from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig
from electrodrive.utils.device import ensure_cuda_available_or_skip


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


def test_step10_backward_compat_generators_init(tmp_path: Path) -> None:
    ensure_cuda_available_or_skip("step10 backward compat init")
    diffusion = DiffusionBasisGenerator(DiffusionGeneratorConfig(k_max=1, n_steps=1))
    assert diffusion is not None

    ckpt_path = tmp_path / "gfn_checkpoint.pt"
    _write_gfn_checkpoint(ckpt_path, spec_dim=4)
    generator = GFlowNetProgramGenerator(checkpoint_path=str(ckpt_path), device=torch.device("cuda"))
    assert generator is not None
