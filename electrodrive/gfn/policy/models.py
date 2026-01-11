"""Policy and logZ networks for factorized GFlowNet sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from electrodrive.gfn.env import ElectrodriveProgramEnv, PartialProgramState
from electrodrive.gfn.dsl.action import Action
from electrodrive.gfn.dsl.tokenize import PAD_TOKEN_ID, TOKEN_MAP, tokenize_program
from electrodrive.utils.device import get_default_device


_FACTOR_NAMES: Tuple[str, ...] = (
    "action_type",
    "subtype",
    "family",
    "conductor",
    "motif",
    "interface",
    "arg0",
    "arg1",
)


@dataclass(frozen=True)
class ActionFactorSizes:
    """Vocabulary sizes for factorized policy heads."""

    action_type: int
    subtype: int
    family: int
    conductor: int
    motif: int
    interface: int
    arg0: int
    arg1: int


@dataclass
class ActionFactorTable:
    """Precomputed factor tables for masking and sampling."""

    action_table: torch.Tensor
    factor_values: Mapping[str, torch.Tensor]
    factor_onehot: Mapping[str, torch.Tensor]
    vocab_sizes: Mapping[str, int]
    device: torch.device


@dataclass(frozen=True)
class PolicyOutputs:
    """Container for factorized policy logits."""

    logits_action_type: torch.Tensor
    logits_subtype: torch.Tensor
    logits_family: torch.Tensor
    logits_conductor: torch.Tensor
    logits_motif: torch.Tensor
    logits_interface: torch.Tensor
    logits_arg0: torch.Tensor
    logits_arg1: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return logits indexed by factor name."""
        return {
            "action_type": self.logits_action_type,
            "subtype": self.logits_subtype,
            "family": self.logits_family,
            "conductor": self.logits_conductor,
            "motif": self.logits_motif,
            "interface": self.logits_interface,
            "arg0": self.logits_arg0,
            "arg1": self.logits_arg1,
        }


@dataclass(frozen=True)
class PolicySample:
    """Sampled actions and log-probabilities from the forward policy."""

    actions: List[Action]
    encoded_actions: torch.Tensor
    logpf: torch.Tensor
    action_indices: torch.Tensor


@dataclass(frozen=True)
class PolicyNetConfig:
    """Configuration for the factorized policy network."""

    spec_dim: int
    token_embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    max_seq_len: int = 64
    token_vocab_size: int = 0


class PolicyNet(nn.Module):
    """Forward policy network with factorized action heads."""

    def __init__(
        self,
        config: PolicyNetConfig,
        factor_sizes: ActionFactorSizes,
        *,
        device: Optional[torch.device] = None,
        token_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.factor_sizes = factor_sizes
        self.device = device or get_default_device()
        self.pad_token_id = int(PAD_TOKEN_ID)
        vocab_override = int(token_vocab_size or 0)
        config_vocab = int(getattr(config, "token_vocab_size", 0))
        token_vocab = vocab_override if vocab_override > 0 else config_vocab
        if token_vocab <= 0:
            token_vocab = int(max(TOKEN_MAP.values())) + 1

        self.token_embed = nn.Embedding(token_vocab, config.token_embed_dim)
        self.state_proj = nn.Linear(config.token_embed_dim, config.hidden_dim)
        self.spec_proj = nn.Linear(config.spec_dim, config.hidden_dim)
        self.shared = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        self.head_action_type = nn.Linear(config.hidden_dim, factor_sizes.action_type)
        self.head_subtype = nn.Linear(config.hidden_dim, factor_sizes.subtype)
        self.head_family = nn.Linear(config.hidden_dim, factor_sizes.family)
        self.head_conductor = nn.Linear(config.hidden_dim, factor_sizes.conductor)
        self.head_motif = nn.Linear(config.hidden_dim, factor_sizes.motif)
        self.head_interface = nn.Linear(config.hidden_dim, factor_sizes.interface)
        self.head_arg0 = nn.Linear(config.hidden_dim, factor_sizes.arg0)
        self.head_arg1 = nn.Linear(config.hidden_dim, factor_sizes.arg1)
        self.to(self.device)

    def forward(self, spec_embedding: torch.Tensor, token_ids: torch.Tensor) -> PolicyOutputs:
        """Compute factorized logits for the provided batch."""
        if spec_embedding.device != self.device:
            spec_embedding = spec_embedding.to(self.device)
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)

        token_emb = self.token_embed(token_ids)
        mask = token_ids != self.pad_token_id
        mask_f = mask.to(token_emb.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (token_emb * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        state_feat = self.state_proj(pooled)
        spec_feat = self.spec_proj(spec_embedding)
        combined = torch.cat([state_feat, spec_feat], dim=-1)
        hidden = self.shared(combined)
        return PolicyOutputs(
            logits_action_type=self.head_action_type(hidden),
            logits_subtype=self.head_subtype(hidden),
            logits_family=self.head_family(hidden),
            logits_conductor=self.head_conductor(hidden),
            logits_motif=self.head_motif(hidden),
            logits_interface=self.head_interface(hidden),
            logits_arg0=self.head_arg0(hidden),
            logits_arg1=self.head_arg1(hidden),
        )


class LogZNet(nn.Module):
    """Conditional logZ network for GFlowNet training."""

    def __init__(self, spec_dim: int, hidden_dim: int = 128, *, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or get_default_device()
        self.net = nn.Sequential(
            nn.Linear(spec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)

    def forward(self, spec_embedding: torch.Tensor) -> torch.Tensor:
        """Compute logZ for each spec embedding in the batch."""
        if spec_embedding.device != self.device:
            spec_embedding = spec_embedding.to(self.device)
        return self.net(spec_embedding).squeeze(-1)


def build_action_factor_table(
    env: ElectrodriveProgramEnv, *, device: Optional[torch.device] = None
) -> ActionFactorTable:
    """Precompute factor values and one-hot masks for an environment."""
    device = device or env.device
    encoded = env._encoded_actions.to(device=device, dtype=torch.int64)
    if encoded.ndim != 2 or encoded.shape[1] < env.ACTION_ENCODING_SIZE:
        raise ValueError("Encoded actions must have shape [num_actions, ACTION_ENCODING_SIZE]")

    cols = [
        encoded[:, 0],
        encoded[:, 1],
        encoded[:, 3],
        encoded[:, 2],
        encoded[:, 4],
        encoded[:, 5],
        encoded[:, 6],
        encoded[:, 7],
    ]
    action_table = torch.stack(cols, dim=1)
    action_table[:, 1:] = action_table[:, 1:] + 1

    vocab_sizes: Dict[str, int] = {"action_type": len(env.action_type_to_id)}
    for name, col in zip(_FACTOR_NAMES[1:], cols[1:]):
        max_val = int(col.max().item())
        vocab_sizes[name] = 1 if max_val < 0 else max_val + 2

    factor_values = {name: action_table[:, idx] for idx, name in enumerate(_FACTOR_NAMES)}
    factor_onehot = {
        name: F.one_hot(values, num_classes=vocab_sizes[name]).to(torch.bool)
        for name, values in factor_values.items()
    }
    return ActionFactorTable(
        action_table=action_table,
        factor_values=factor_values,
        factor_onehot=factor_onehot,
        vocab_sizes=vocab_sizes,
        device=device,
    )


def action_factor_sizes_from_table(table: ActionFactorTable) -> ActionFactorSizes:
    """Build ActionFactorSizes from a precomputed table."""
    return ActionFactorSizes(
        action_type=table.vocab_sizes["action_type"],
        subtype=table.vocab_sizes["subtype"],
        family=table.vocab_sizes["family"],
        conductor=table.vocab_sizes["conductor"],
        motif=table.vocab_sizes["motif"],
        interface=table.vocab_sizes["interface"],
        arg0=table.vocab_sizes["arg0"],
        arg1=table.vocab_sizes["arg1"],
    )


def sample_actions(
    policy: PolicyNet,
    env: ElectrodriveProgramEnv,
    states: Sequence[PartialProgramState],
    spec_embedding: torch.Tensor,
    *,
    temperature: float = 1.0,
    factor_table: Optional[ActionFactorTable] = None,
    generator: Optional[torch.Generator] = None,
) -> PolicySample:
    """Sample factorized actions for a batch of states."""
    if not states:
        raise ValueError("States batch cannot be empty")
    device = policy.device
    if factor_table is None or factor_table.device != device:
        factor_table = build_action_factor_table(env, device=device)

    token_ids = _stack_state_tokens(states, env, device)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    autocast_dtype = amp_dtype if amp_enabled else torch.float32
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        outputs = policy(spec_embedding=spec_embedding, token_ids=token_ids)
    mask_batch, _ = env.get_action_mask_batch(states)
    mask_batch = mask_batch.to(device)

    logits_by_factor = outputs.as_dict()
    candidate_mask = mask_batch.clone()
    logpf = torch.zeros((mask_batch.shape[0],), device=device)
    sampled_values: Dict[str, torch.Tensor] = {}

    for name in _FACTOR_NAMES:
        logits = logits_by_factor[name]
        allowed = _allowed_values(candidate_mask, factor_table.factor_onehot[name])
        sample, logprob = _sample_from_logits(logits, allowed, temperature, generator)
        sampled_values[name] = sample
        logpf = logpf + logprob
        values = factor_table.factor_values[name]
        candidate_mask = candidate_mask & (values.unsqueeze(0) == sample.unsqueeze(1))

    if not torch.all(candidate_mask.any(dim=1)):
        raise RuntimeError("No legal action remaining after factorized sampling")
    action_indices = candidate_mask.to(torch.int64).argmax(dim=1)
    encoded_actions = env._encoded_actions.to(device=device).index_select(0, action_indices)
    action_list = [env.actions[idx] for idx in action_indices.to("cpu").tolist()]
    return PolicySample(
        actions=action_list,
        encoded_actions=encoded_actions,
        logpf=logpf,
        action_indices=action_indices,
    )


def _stack_state_tokens(
    states: Sequence[PartialProgramState],
    env: ElectrodriveProgramEnv,
    device: torch.device,
) -> torch.Tensor:
    tokens: List[torch.Tensor] = []
    max_len = env.max_length
    for state in states:
        token_ids = state.ast_token_ids
        if token_ids is None:
            token_ids = tokenize_program(state.program, max_len=max_len, device=device, grammar=env.grammar)
        if token_ids.device != device:
            token_ids = token_ids.to(device)
        tokens.append(token_ids)
    return torch.stack(tokens, dim=0)


def _allowed_values(candidate_mask: torch.Tensor, factor_onehot: torch.Tensor) -> torch.Tensor:
    return (candidate_mask.unsqueeze(-1) & factor_onehot.unsqueeze(0)).any(dim=1)


def _sample_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("Logits must be 2D [batch, vocab]")
    if not torch.all(mask.any(dim=1)):
        raise RuntimeError("Masked logits contain empty rows")
    temp = float(temperature)
    if temp <= 0.0:
        temp = 1e-3
    scaled = logits / temp
    neg_inf = torch.finfo(scaled.dtype).min
    masked_logits = torch.where(mask, scaled, torch.tensor(neg_inf, device=scaled.device, dtype=scaled.dtype))
    _ = generator
    dist = torch.distributions.Categorical(logits=masked_logits)
    sample = dist.sample()
    logprob = dist.log_prob(sample)
    return sample, logprob


__all__ = [
    "ActionFactorSizes",
    "ActionFactorTable",
    "PolicyOutputs",
    "PolicySample",
    "PolicyNet",
    "PolicyNetConfig",
    "LogZNet",
    "build_action_factor_table",
    "action_factor_sizes_from_table",
    "sample_actions",
]
