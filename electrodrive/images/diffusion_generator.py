from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from electrodrive.images.basis import (
    BasisGenerator,
    ImageBasisElement,
    PointChargeBasis,
    annotate_group_info,
    ChargeNodeInfo,
    CondNodeInfo,
)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding used in diffusion models."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(start=0, end=half, device=device, dtype=torch.float32)
        / float(half)
    )
    args = t.float().unsqueeze(-1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _default_type_names() -> Tuple[str, ...]:
    """Stable default basis families for diffusion slots."""
    return ("learned_point", "axis_point", "point")


@dataclass
class DiffusionGeneratorConfig:
    """Lightweight hyperparameters for the diffusion-based BasisGenerator."""

    k_max: int = 32
    type_names: Tuple[str, ...] = field(default_factory=_default_type_names)
    hidden_dim: int = 128
    time_emb_dim: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_steps: int = 32
    beta_min: float = 1e-4
    beta_max: float = 2e-2
    sigma_min: float = 0.0
    use_cosine_schedule: bool = False
    # Local frame mapping is currently identity; extend in later passes.
    local_frame: str = "identity"


class _DiffusionSetDenoiser(nn.Module):
    """Set-transformer style denoiser for position + type noise."""

    def __init__(self, cfg: DiffusionGeneratorConfig, n_types: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        self.slot_embed = nn.Linear(3 + n_types, H)
        self.cond_proj = nn.LazyLinear(H)
        self.time_proj = nn.Linear(cfg.time_emb_dim, H)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            embed_dim=H,
                            num_heads=cfg.n_heads,
                            batch_first=True,
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=H,
                            num_heads=cfg.n_heads,
                            batch_first=True,
                        ),
                        "mlp": nn.Sequential(
                            nn.Linear(H, 2 * H),
                            nn.ReLU(),
                            nn.Linear(2 * H, H),
                        ),
                        "norm1": nn.LayerNorm(H),
                        "norm2": nn.LayerNorm(H),
                        "norm3": nn.LayerNorm(H),
                    }
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.head_pos = nn.Linear(H, 3)
        self.head_type = nn.Linear(H, n_types)

    def forward(
        self,
        x_noisy: torch.Tensor,
        logits_noisy: torch.Tensor,
        cond_vec: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_noisy : [B, K, 3]
        logits_noisy : [B, K, T]
        cond_vec : [B, H_c]
        t_emb : [B, H_t]
        mask : [B, K] or None
        """
        B, K, _ = x_noisy.shape
        H = self.cfg.hidden_dim

        slot_in = torch.cat([x_noisy, logits_noisy], dim=-1)
        h = self.slot_embed(slot_in)
        cond = self.cond_proj(cond_vec).unsqueeze(1)  # [B,1,H]
        t_proj = self.time_proj(t_emb).unsqueeze(1)
        h = h + t_proj

        attn_mask = None
        if mask is not None:
            # MultiheadAttention expects True for positions to mask.
            attn_mask = ~mask.bool()

        for blk in self.layers:
            h_norm = blk["norm1"](h)
            h_attn, _ = blk["self_attn"](h_norm, h_norm, h_norm, key_padding_mask=attn_mask)
            h = h + h_attn

            h_norm = blk["norm2"](h)
            h_cross, _ = blk["cross_attn"](h_norm, cond, cond, key_padding_mask=None)
            h = h + h_cross

            h_norm = blk["norm3"](h)
            h = h + blk["mlp"](h_norm)

        eps_x = self.head_pos(h)
        eps_t = self.head_type(h)
        return eps_x, eps_t


class DiffusionBasisGenerator(BasisGenerator):
    """Diffusion-based candidate proposer for image basis elements."""

    def __init__(
        self,
        cfg: Optional[DiffusionGeneratorConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DiffusionGeneratorConfig()
        self.type_names = list(self.cfg.type_names)
        self.denoiser = _DiffusionSetDenoiser(
            cfg=self.cfg,
            n_types=len(self.type_names),
        )
        self.register_buffer("betas", self._make_beta_schedule(), persistent=False)
        self.register_buffer("alphas", 1.0 - self.betas, persistent=False)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - self.betas, dim=0), persistent=False)

    # ------------------------------------------------------------------ #
    # Schedules and sampling
    # ------------------------------------------------------------------ #
    def _make_beta_schedule(self) -> torch.Tensor:
        if self.cfg.use_cosine_schedule:
            steps = self.cfg.n_steps
            t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
            s = 0.008
            alphas_cum = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cum = alphas_cum / alphas_cum[0]
            betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
            return betas.clamp(1e-5, 0.999)
        # Linear schedule fallback.
        return torch.linspace(self.cfg.beta_min, self.cfg.beta_max, self.cfg.n_steps, dtype=torch.float32)

    def _reverse_step(
        self,
        x_t: torch.Tensor,
        logits_t: torch.Tensor,
        cond_vec: torch.Tensor,
        t_step: int,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_t.shape[0]
        t = torch.full((B,), float(t_step), device=x_t.device)
        t_emb = _sinusoidal_time_embedding(t, self.cfg.time_emb_dim)
        eps_x, eps_t = self.denoiser(x_t, logits_t, cond_vec, t_emb, mask)

        beta_t = self.betas[t_step]
        alpha_t = self.alphas[t_step]
        alpha_bar_t = self.alpha_bars[t_step]
        alpha_bar_prev = self.alpha_bars[t_step - 1] if t_step > 0 else torch.tensor(1.0, device=x_t.device)
        beta_t = beta_t.to(device=x_t.device, dtype=x_t.dtype)
        alpha_t = alpha_t.to(device=x_t.device, dtype=x_t.dtype)
        alpha_bar_t = alpha_bar_t.to(device=x_t.device, dtype=x_t.dtype)
        alpha_bar_prev = alpha_bar_prev.to(device=x_t.device, dtype=x_t.dtype)

        # DDPM update (simplified, shared for positions + logits).
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(torch.clamp(alpha_bar_t, min=1e-5))

        x_prev = coef1 * (x_t - coef2 * eps_x)
        logits_prev = coef1 * (logits_t - coef2 * eps_t)

        if t_step > 0:
            sigma_t = torch.sqrt(torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t, min=0.0))
            noise_x = torch.randn_like(x_t) * sigma_t
            noise_t = torch.randn_like(logits_t) * sigma_t
            x_prev = x_prev + noise_x
            logits_prev = logits_prev + noise_t

        return x_prev, logits_prev

    def sample(
        self,
        cond_vec: torch.Tensor,
        *,
        mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a truncated reverse diffusion process."""
        K = self.cfg.k_max
        T = len(self.type_names)
        x_t = torch.randn(n_samples, K, 3, device=device, dtype=dtype)
        logits_t = torch.randn(n_samples, K, T, device=device, dtype=dtype)

        for step in reversed(range(self.cfg.n_steps)):
            x_t, logits_t = self._reverse_step(x_t, logits_t, cond_vec, step, mask)

        mask_tensor = mask if mask is not None else torch.ones(n_samples, K, device=device, dtype=torch.bool)
        return x_t, logits_t, mask_tensor

    # ------------------------------------------------------------------ #
    # BasisGenerator API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        z_global: torch.Tensor,
        charge_nodes: Sequence[ChargeNodeInfo],
        conductor_nodes: Sequence[CondNodeInfo],
        n_candidates: int,
    ) -> List[ImageBasisElement]:
        device = z_global.device
        dtype = z_global.dtype
        if n_candidates <= 0:
            return []

        # Aggregate charge embeddings.
        if charge_nodes:
            emb_stack = torch.stack([c.embedding for c in charge_nodes], dim=0)
            charge_mean = emb_stack.mean(dim=0)
        else:
            charge_mean = torch.zeros_like(z_global)

        cond_vec = torch.cat([z_global, charge_mean], dim=-1).unsqueeze(0)
        mask = torch.ones(1, self.cfg.k_max, device=device, dtype=torch.bool)

        with torch.no_grad():
            x_samples, logits_samples, mask_tensor = self.sample(
                cond_vec=cond_vec,
                mask=mask,
                device=device,
                dtype=dtype,
                n_samples=1,
            )

        x_final = x_samples[0]
        logits_final = logits_samples[0]
        mask_final = mask_tensor[0]

        elems: List[ImageBasisElement] = []
        for j in range(min(self.cfg.k_max, n_candidates)):
            if not bool(mask_final[j].item()):
                continue
            pos_local = x_final[j]
            if not torch.isfinite(pos_local).all():
                continue
            type_idx = int(torch.argmax(logits_final[j]).item())
            type_name = self.type_names[type_idx % len(self.type_names)]
            elem = self._make_element(type_name, pos_local, conductor_nodes)
            if elem is None:
                continue
            annotate_group_info(
                elem,
                conductor_id=0,
                family_name=elem.type,
                motif_index=j + 1,
            )
            elems.append(elem)
        return elems[:n_candidates]

    def _make_element(
        self,
        type_name: str,
        pos_local: torch.Tensor,
        conductor_nodes: Sequence[CondNodeInfo],
    ) -> Optional[ImageBasisElement]:
        # Local-to-world mapping is currently identity; TODO: incorporate geometry frames.
        pos_world = pos_local

        # Optionally bias toward first conductor center if available.
        if conductor_nodes:
            center = conductor_nodes[0].center
            if torch.isfinite(center).all():
                pos_world = pos_world + center

        if type_name in {"point", "learned_point", "axis_point"}:
            return PointChargeBasis({"position": pos_world}, type_name=type_name if type_name != "point" else "point")

        # Fallback to a point-like element for unknown types to keep sampling robust.
        return PointChargeBasis({"position": pos_world}, type_name="learned_point")
