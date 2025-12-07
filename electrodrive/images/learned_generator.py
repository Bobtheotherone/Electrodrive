from __future__ import annotations

from typing import List, Tuple

import math
import torch
import torch.nn as nn

from electrodrive.images.basis import (
    BasisGenerator,
    ChargeNodeInfo,
    CondNodeInfo,
    PointChargeBasis,
)
from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    return nn.Sequential(*layers)


class SimpleGeoEncoder(nn.Module):
    """
    Minimal geometry encoder that lifts raw spec fields into latent vectors.

    This is intentionally lightweight and deterministic so it can run inside
    tests and CLI paths without external dependencies or training.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Charge: pos(3) + q(1)
        self.charge_mlp = _mlp(4, hidden_dim, latent_dim)
        # Conductor: center(3) + radius-ish(1) + type_id(1)
        self.cond_mlp = _mlp(5, hidden_dim, latent_dim)
        # Global mixes charge + conductor summaries.
        self.global_mlp = _mlp(2 * latent_dim, hidden_dim, latent_dim)

    def _type_id(self, t: str | None) -> int:
        if t == "sphere":
            return 1
        if t in ("plane", "ground_plane"):
            return 2
        if t in ("torus", "toroid"):
            return 3
        return 0

    def encode(
        self,
        spec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[ChargeNodeInfo], List[CondNodeInfo]]:
        self.to(device=device)
        charge_nodes: List[ChargeNodeInfo] = []
        cond_nodes: List[CondNodeInfo] = []

        for ch in getattr(spec, "charges", []):
            if ch.get("type") != "point":
                continue
            pos_raw = ch.get("pos", None)
            if pos_raw is None:
                continue
            pos = torch.tensor(pos_raw, device=device, dtype=dtype).view(3)
            q_val = float(ch.get("q", 0.0))
            inp = torch.cat(
                [
                    pos,
                    torch.tensor([q_val], device=device, dtype=dtype),
                ],
                dim=0,
            )
            emb = self.charge_mlp(inp)
            charge_nodes.append(ChargeNodeInfo(pos=pos, embedding=emb, q=q_val))

        for cond in getattr(spec, "conductors", []):
            t = cond.get("type")
            center = torch.tensor(cond.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype).view(3)
            radius = float(cond.get("radius", cond.get("minor_radius", 1.0)))
            type_id = self._type_id(t)
            inp = torch.cat(
                [
                    center,
                    torch.tensor([radius, float(type_id)], device=device, dtype=dtype),
                ],
                dim=0,
            )
            emb = self.cond_mlp(inp)
            cond_nodes.append(
                CondNodeInfo(
                    center=center,
                    embedding=emb,
                    type_id=type_id,
                    radius=radius if math.isfinite(radius) else None,
                )
            )

        # Aggregate globals (mean-pool) then mix with a small MLP.
        if charge_nodes:
            c_stack = torch.stack([c.embedding for c in charge_nodes], dim=0)
            c_mean = c_stack.mean(dim=0)
        else:
            c_mean = torch.zeros(self.latent_dim, device=device, dtype=dtype)

        if cond_nodes:
            g_stack = torch.stack([g.embedding for g in cond_nodes], dim=0)
            g_mean = g_stack.mean(dim=0)
        else:
            g_mean = torch.zeros(self.latent_dim, device=device, dtype=dtype)

        z_global = self.global_mlp(torch.cat([c_mean, g_mean], dim=-1))
        return z_global.to(dtype=dtype), charge_nodes, cond_nodes


class MLPBasisGenerator(BasisGenerator):
    """
    Simple learned basis generator that emits point-like primitives.

    This is a placeholder for future training; it deterministically maps
    latent geometry embeddings to candidate positions with lightweight MLPs.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        noise_dim: int = 8,
        max_radius_scale: float = 5.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.max_radius_scale = max_radius_scale
        in_dim = 3 * latent_dim + noise_dim
        # Outputs: logits(2), r_int(1), r_ext(1), axial(1), theta(1), normal(3)
        self.head = _mlp(in_dim, hidden_dim, 9)

    def forward(
        self,
        z_global: torch.Tensor,
        charge_nodes: List[ChargeNodeInfo],
        conductor_nodes: List[CondNodeInfo],
        n_candidates: int,
    ) -> List[PointChargeBasis]:
        device = z_global.device
        dtype = z_global.dtype
        if n_candidates <= 0 or not conductor_nodes:
            return []

        # Fallback charge embedding when no charges are present.
        fallback_charge = torch.zeros(self.latent_dim, device=device, dtype=dtype)
        fallback_charge_pos = torch.zeros(3, device=device, dtype=dtype)
        candidates: List[PointChargeBasis] = []

        cond_centers = [c.center for c in conductor_nodes]

        for k in range(n_candidates):
            # Pick nearest charge to nearest conductor center to stabilise.
            cond = conductor_nodes[min(k % len(conductor_nodes), len(conductor_nodes) - 1)]
            if charge_nodes:
                dists = [
                    torch.linalg.norm(ch.pos - cond.center).item()
                    for ch in charge_nodes
                ]
                idx = int(min(range(len(dists)), key=lambda i: dists[i]))
                charge = charge_nodes[idx]
                z_charge = charge.embedding
                charge_pos = charge.pos
            else:
                z_charge = fallback_charge
                charge_pos = fallback_charge_pos

            noise = torch.randn(self.noise_dim, device=device, dtype=dtype)
            z_in = BasisGenerator.combine_latents(z_global, z_charge, cond.embedding, noise)
            out = self.head(z_in)

            logits = out[:2]
            use_interior = int(torch.argmax(logits).item()) == 0
            r_int_raw = out[2]
            r_ext_raw = out[3]
            axial_raw = out[4]
            theta_raw = out[5]
            normal_raw = out[6:9]

            base_radius = cond.radius if cond.radius is not None else 1.0
            r_int = BasisGenerator.map_interior_radius(r_int_raw, base_radius, safety=0.98)
            r_ext = BasisGenerator.map_exterior_radius(r_ext_raw, base_radius, max_scale=self.max_radius_scale)
            r = r_int if use_interior else r_ext

            theta = torch.sigmoid(theta_raw) * (2.0 * torch.pi)
            axial = torch.tanh(axial_raw) * (0.5 * base_radius)

            u_vec, v_vec, n_vec = BasisGenerator.local_frame_from_normal(normal_raw)
            pos_local = (
                r * (torch.cos(theta) * u_vec + torch.sin(theta) * v_vec)
                + axial * n_vec
            )

            pos_world = cond.center.to(device=device, dtype=dtype) + pos_local

            # If the charge is defined, gently bias toward the charge-facing side.
            if cond.radius is not None:
                rel_charge = charge_pos - cond.center
                dist_charge = torch.linalg.norm(rel_charge).item()
                if dist_charge > 1e-9:
                    axis_dir = rel_charge / dist_charge
                    # Project a small component toward the charge-facing hemisphere.
                    bias = 0.15 * base_radius * axis_dir
                    pos_world = pos_world + bias

            if not torch.isfinite(pos_world).all():
                continue

            candidates.append(
                PointChargeBasis(
                    {"position": pos_world},
                    type_name="learned_point",
                )
            )

        return candidates[:n_candidates]
