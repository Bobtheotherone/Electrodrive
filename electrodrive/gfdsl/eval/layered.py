"""Layered-media evaluation helpers for GFDSL."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from electrodrive.gfdsl.eval.kernels_real import coulomb_potential_real


def _resolve_source_xy(ctx: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    extras = getattr(ctx, "extras", {}) or {}
    if "source_xy" in extras:
        xy = torch.as_tensor(extras["source_xy"], device=device, dtype=dtype).view(-1)
        if xy.numel() >= 2:
            return xy[:2]
    if "source_pos" in extras:
        pos = torch.as_tensor(extras["source_pos"], device=device, dtype=dtype).view(-1)
        if pos.numel() >= 2:
            return pos[:2]

    spec = getattr(ctx, "spec", None)
    charges = None
    if spec is not None:
        if hasattr(spec, "charges"):
            charges = getattr(spec, "charges")
        elif isinstance(spec, dict):
            charges = spec.get("charges")
    if charges:
        for ch in charges:
            if isinstance(ch, dict) and ch.get("type") == "point":
                pos = torch.tensor(ch.get("pos", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
                return pos[:2]
    return torch.zeros(2, device=device, dtype=dtype)


def _resolve_interface_z(ctx: Any, meta: dict | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    meta = meta or {}
    if "interface_z" in meta:
        return torch.as_tensor(meta["interface_z"], device=device, dtype=dtype).view(())
    if "z_interface" in meta:
        return torch.as_tensor(meta["z_interface"], device=device, dtype=dtype).view(())

    extras = getattr(ctx, "extras", {}) or {}
    if "interface_z" in extras:
        return torch.as_tensor(extras["interface_z"], device=device, dtype=dtype).view(())
    if "z_interface" in extras:
        return torch.as_tensor(extras["z_interface"], device=device, dtype=dtype).view(())

    return torch.zeros((), device=device, dtype=dtype)


def interface_pole_columns(
    X: torch.Tensor,
    *,
    src_xy: torch.Tensor,
    z_interface: torch.Tensor,
    depths: torch.Tensor,
    residues: torch.Tensor,
) -> torch.Tensor:
    depths = depths.view(-1)
    residues = residues.view(-1)
    if residues.numel() == 1 and depths.numel() > 1:
        residues = residues.expand_as(depths)
    if depths.numel() == 1 and residues.numel() > 1:
        depths = depths.expand_as(residues)
    if depths.numel() != residues.numel():
        raise ValueError("InterfacePoleNode depths/residues must have matching lengths")

    z_pos = z_interface - depths
    x = src_xy[0].expand_as(z_pos)
    y = src_xy[1].expand_as(z_pos)
    pos = torch.stack((x, y, z_pos), dim=-1)
    phi = coulomb_potential_real(X, pos)
    return phi * residues.view(1, -1)


def branch_cut_exp_sum_columns(
    X: torch.Tensor,
    *,
    src_xy: torch.Tensor,
    z_interface: torch.Tensor,
    depths: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    depths = depths.view(-1)
    weights = weights.view(-1)
    if weights.numel() == 1 and depths.numel() > 1:
        weights = weights.expand_as(depths)
    if depths.numel() == 1 and weights.numel() > 1:
        depths = depths.expand_as(weights)
    if depths.numel() != weights.numel():
        raise ValueError("BranchCutApproxNode depths/weights must have matching lengths")

    z_pos = z_interface - depths
    x = src_xy[0].expand_as(z_pos)
    y = src_xy[1].expand_as(z_pos)
    pos = torch.stack((x, y, z_pos), dim=-1)
    phi = coulomb_potential_real(X, pos)
    return phi * weights.view(1, -1)


def resolve_layered_frame(
    ctx: Any,
    meta: dict | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_xy = _resolve_source_xy(ctx, device=device, dtype=dtype)
    z_interface = _resolve_interface_z(ctx, meta, device=device, dtype=dtype)
    return src_xy, z_interface
