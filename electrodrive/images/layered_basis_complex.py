"""GPU-first layered planar basis helpers (experimental, opt-in).

This module assembles structured candidate positions for three-layer
dielectric stacks. It performs only lightweight CUDA allocations and
fails fast if CUDA is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from electrodrive.orchestration.parser import CanonicalSpec


@dataclass
class LayeredBasisCandidate:
    """Structured candidate description for layered stacks."""

    position: torch.Tensor
    family: str
    motif_index: int
    conductor_id: int


def _ensure_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(f"Layered basis requires CUDA device (got {device})")
    if not torch.cuda.is_available():
        raise RuntimeError("Layered basis requires CUDA; CUDA not available.")


def _select_slab_layer(dielectrics: Sequence[dict]) -> Optional[dict]:
    """Pick the finite slab (min positive thickness)."""
    slab_layer = None
    slab_thickness = None
    for layer in dielectrics:
        z_min = layer.get("z_min")
        z_max = layer.get("z_max")
        if z_min is None or z_max is None:
            continue
        try:
            thickness = float(z_max) - float(z_min)
        except Exception:
            continue
        if thickness <= 0:
            continue
        if slab_thickness is None or thickness < slab_thickness:
            slab_thickness = thickness
            slab_layer = layer
    return slab_layer


def _layer_region_id(z_val: float, top_z: float, bottom_z: float, tol: float = 1e-9) -> int:
    if z_val > top_z + tol:
        return 0  # region1 (top half-space)
    if z_val < bottom_z - tol:
        return 2  # region3 (bottom half-space)
    return 1  # slab interior


def generate_three_layer_candidates(
    spec: CanonicalSpec,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    max_candidates: int = 12,
) -> List[LayeredBasisCandidate]:
    """Generate slab-aware candidate locations for three-layer stacks."""
    dev = torch.device(device)
    _ensure_cuda(dev)
    slab = _select_slab_layer(getattr(spec, "dielectrics", []) or [])
    if slab is None:
        return []
    top_z = float(slab.get("z_max", 0.0))
    bottom_z = float(slab.get("z_min", 0.0))
    h = float(top_z - bottom_z)
    if h <= 0.0:
        return []

    charges = [
        c for c in getattr(spec, "charges", []) if c.get("type") == "point" and c.get("pos") is not None
    ]
    if not charges:
        return []

    out: List[LayeredBasisCandidate] = []

    for charge in charges:
        try:
            z0 = float(charge["pos"][2])
        except Exception:
            continue

        try:
            xy = (float(charge["pos"][0]), float(charge["pos"][1]))
        except Exception:
            xy = (0.0, 0.0)

        def _append(z_val: float, family: str, motif: int) -> None:
            nonlocal out
            if len(out) >= max_candidates:
                return
            pos = torch.tensor([xy[0], xy[1], z_val], device=dev, dtype=dtype)
            out.append(
                LayeredBasisCandidate(
                    position=pos,
                    family=family,
                    motif_index=motif,
                    conductor_id=_layer_region_id(z_val, top_z, bottom_z),
                )
            )

        # Mirrors across the top interface and first ladder bounce.
        mirror_z = 2.0 * top_z - z0
        _append(mirror_z, "three_layer_complex_mirror", 0)
        _append(mirror_z - 2.0 * h, "three_layer_complex_mirror", 1)

        # Slab interior anchors (quarter / three-quarter depth).
        for idx, frac in enumerate((0.25, 0.75)):
            _append(top_z - frac * h, "three_layer_complex_slab", idx)

        # Evanescent tails below the slab to mimic guided decay.
        for idx, alpha in enumerate((0.5, 1.5, 3.0)):
            _append(bottom_z - alpha * h, "three_layer_complex_tail", idx)

    return out[:max_candidates]
