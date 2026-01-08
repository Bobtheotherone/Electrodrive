from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch


def _spec_field(spec: object, name: str) -> list[dict]:
    if isinstance(spec, dict):
        val = spec.get(name, [])
    else:
        val = getattr(spec, name, [])  # type: ignore[attr-defined]
    return list(val or [])


def _charge_positions(spec: object, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    charges = _spec_field(spec, "charges")
    positions: List[torch.Tensor] = []
    for ch in charges:
        if ch.get("type") != "point":
            continue
        pos_raw = ch.get("pos")
        if not isinstance(pos_raw, (list, tuple)) or len(pos_raw) != 3:
            continue
        positions.append(torch.tensor(pos_raw, device=device, dtype=dtype).view(3))
    return positions


def parse_layered_interfaces(spec: object) -> List[float]:
    dielectrics = _spec_field(spec, "dielectrics")
    planes: List[float] = []
    for layer in dielectrics:
        z_min = layer.get("z_min", None)
        z_max = layer.get("z_max", None)
        try:
            if z_min is not None and math.isfinite(float(z_min)):
                planes.append(float(z_min))
        except Exception:
            pass
        try:
            if z_max is not None and math.isfinite(float(z_max)):
                planes.append(float(z_max))
        except Exception:
            pass
    return sorted({float(z) for z in planes})


def _layered_bounds(spec: object, domain_scale: float) -> Tuple[float, float]:
    interfaces = parse_layered_interfaces(spec)
    if interfaces:
        z_min = min(interfaces) - domain_scale
        z_max = max(interfaces) + domain_scale
    else:
        z_min, z_max = -domain_scale, domain_scale
    for ch in _spec_field(spec, "charges"):
        if ch.get("type") != "point":
            continue
        try:
            zq = float(ch.get("pos", [0.0, 0.0, 0.0])[2])
        except Exception:
            continue
        z_min = min(z_min, zq - domain_scale)
        z_max = max(z_max, zq + domain_scale)
    if z_min == z_max:
        z_min, z_max = z_min - 1.0, z_max + 1.0
    return float(z_min), float(z_max)


def _allocate_counts(lengths: List[float], n: int) -> List[int]:
    if not lengths or n <= 0:
        return []
    total = sum(lengths)
    if total <= 0:
        return [n] + [0] * (len(lengths) - 1)
    raw = [length / total * n for length in lengths]
    counts = [int(math.floor(val)) for val in raw]
    remainder = n - sum(counts)
    if remainder > 0:
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def _filter_exclusion(
    pts: torch.Tensor,
    charge_pos: List[torch.Tensor],
    exclusion_radius: float,
) -> torch.Tensor:
    if not charge_pos or exclusion_radius <= 0.0 or pts.numel() == 0:
        return pts
    charge_mat = torch.stack(charge_pos, dim=0)
    dists = torch.cdist(pts, charge_mat)
    mask = torch.all(dists > exclusion_radius, dim=1)
    return pts[mask]


def _sample_band_points(
    interfaces: List[float],
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    domain_scale: float,
    seed: int,
    band: float,
) -> torch.Tensor:
    if n <= 0 or not interfaces:
        return torch.empty((0, 3), device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    idx = torch.randint(0, len(interfaces), (n,), generator=gen, device=device)
    z_vals = torch.tensor(interfaces, device=device, dtype=dtype)[idx]
    offsets = (torch.rand(n, generator=gen, device=device, dtype=dtype) * 2.0 - 1.0) * band
    z = z_vals + offsets
    xy = (torch.rand((n, 2), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
    return torch.stack([xy[:, 0], xy[:, 1], z], dim=1)


def sample_layered_interior(
    spec: object,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    exclusion_radius: float,
    interface_band: float,
    domain_scale: float,
) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)

    interfaces = parse_layered_interfaces(spec)
    z_min, z_max = _layered_bounds(spec, domain_scale)
    boundaries = [z_min] + interfaces + [z_max]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    lengths = [max(1e-6, abs(b - a)) for a, b in segments]
    counts = _allocate_counts(lengths, n)

    charge_pos = _charge_positions(spec, device, dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    out: List[torch.Tensor] = []
    needed = n
    attempts = 0
    while needed > 0 and attempts < 6:
        attempts += 1
        batch: List[torch.Tensor] = []
        for (z0, z1), count in zip(segments, counts):
            if count <= 0:
                continue
            z_lo = min(z0, z1)
            z_hi = max(z0, z1)
            z = torch.rand((count,), generator=gen, device=device, dtype=dtype) * (z_hi - z_lo) + z_lo
            xy = (torch.rand((count, 2), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
            batch.append(torch.stack([xy[:, 0], xy[:, 1], z], dim=1))
        if not batch:
            break
        pts = torch.cat(batch, dim=0)
        pts = _filter_exclusion(pts, charge_pos, exclusion_radius)
        if pts.numel() == 0:
            continue
        take = min(needed, pts.shape[0])
        out.append(pts[:take])
        needed -= take

    if needed > 0:
        pad = (torch.rand((needed * 2, 3), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
        pad[:, 2] = torch.rand((needed * 2,), generator=gen, device=device, dtype=dtype) * (z_max - z_min) + z_min
        pad = _filter_exclusion(pad, charge_pos, exclusion_radius)
        if pad.numel() > 0:
            out.append(pad[:needed])

    pts = torch.cat(out, dim=0) if out else torch.empty((0, 3), device=device, dtype=dtype)
    if pts.shape[0] < n:
        remain = n - pts.shape[0]
        extra_attempts = 0
        extra_out: List[torch.Tensor] = []
        while remain > 0 and extra_attempts < 4:
            extra_attempts += 1
            pad = (torch.rand((remain * 2, 3), generator=gen, device=device, dtype=dtype) - 0.5) * 2.0 * domain_scale
            pad[:, 2] = torch.rand((remain * 2,), generator=gen, device=device, dtype=dtype) * (z_max - z_min) + z_min
            pad = _filter_exclusion(pad, charge_pos, exclusion_radius)
            if pad.numel() == 0:
                continue
            take = min(remain, pad.shape[0])
            extra_out.append(pad[:take])
            remain -= take
        if extra_out:
            pts = torch.cat([pts] + extra_out, dim=0)

    if interface_band > 0.0 and interfaces and pts.numel() > 0:
        z = pts[:, 2]
        planes = torch.tensor(interfaces, device=device, dtype=dtype)
        dists = torch.abs(z[:, None] - planes[None, :])
        band_mask = torch.any(dists < interface_band, dim=1)
        target = max(1, n // 10)
        band_count = int(band_mask.sum().item())
        if band_count < target:
            need = target - band_count
            band_pts = _sample_band_points(
                interfaces,
                need,
                device=device,
                dtype=dtype,
                domain_scale=domain_scale,
                seed=seed + 17,
                band=interface_band,
            )
            band_pts = _filter_exclusion(band_pts, charge_pos, exclusion_radius)
            if band_pts.shape[0] > 0:
                replace_idx = torch.where(~band_mask)[0][: band_pts.shape[0]]
                pts[replace_idx] = band_pts[: replace_idx.shape[0]]

    return pts[:n].contiguous()


def sample_layered_interface_pairs(
    spec: object,
    n_xy: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    delta: float,
    domain_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    interfaces = parse_layered_interfaces(spec)
    if not interfaces or n_xy <= 0:
        empty = torch.empty((0, 3), device=device, dtype=dtype)
        return empty, empty
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    xy = engine.draw(n_xy * len(interfaces)).to(device=device, dtype=dtype)
    xy = (xy - 0.5) * 2.0 * domain_scale
    pts_up = []
    pts_dn = []
    idx = 0
    for z_val in interfaces:
        chunk = xy[idx : idx + n_xy]
        idx += n_xy
        z_up = torch.full((n_xy,), z_val + delta, device=device, dtype=dtype)
        z_dn = torch.full((n_xy,), z_val - delta, device=device, dtype=dtype)
        pts_up.append(torch.stack([chunk[:, 0], chunk[:, 1], z_up], dim=1))
        pts_dn.append(torch.stack([chunk[:, 0], chunk[:, 1], z_dn], dim=1))
    return torch.cat(pts_up, dim=0), torch.cat(pts_dn, dim=0)


__all__ = [
    "parse_layered_interfaces",
    "sample_layered_interior",
    "sample_layered_interface_pairs",
]
