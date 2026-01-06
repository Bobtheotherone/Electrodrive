from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


EvalFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class BoundarySpec:
    """Boundary specification for residual checks."""

    kind: str = "plane"
    bc_type: str = "dirichlet"
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.0


@dataclass
class GreenCheckConfig:
    n_samples: int = 256
    domain_radius: float = 1.0
    near_radius: float = 1e-3
    far_radius: float = 5.0
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    seed: int = 1234
    eps: float = 1e-12
    fd_step: float = 1e-3


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rand_uniform(
    n: int,
    radius: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return (torch.rand(n, 3, device=device, dtype=dtype, generator=generator) * 2.0 - 1.0) * radius


def sample_region_pairs(
    cfg: GreenCheckConfig,
    *,
    boundary: Optional[BoundarySpec] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    device = _resolve_device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))

    n = int(cfg.n_samples)
    base = _rand_uniform(
        n,
        cfg.domain_radius,
        device=device,
        dtype=cfg.dtype,
        generator=gen,
    )
    offsets = torch.randn(n, 3, device=device, dtype=cfg.dtype, generator=gen)
    offsets = offsets / torch.linalg.norm(offsets, dim=1, keepdim=True).clamp_min(cfg.eps)
    offsets = offsets * float(cfg.near_radius)
    near_x = base + offsets
    near_y = base

    far_dir = torch.randn(n, 3, device=device, dtype=cfg.dtype, generator=gen)
    far_dir = far_dir / torch.linalg.norm(far_dir, dim=1, keepdim=True).clamp_min(cfg.eps)
    far_x = _rand_uniform(
        n,
        cfg.domain_radius,
        device=device,
        dtype=cfg.dtype,
        generator=gen,
    )
    far_y = far_x + far_dir * float(cfg.far_radius)

    pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "near": (near_x, near_y),
        "far": (far_x, far_y),
    }

    if boundary is not None:
        if boundary.kind != "plane":
            raise ValueError(f"Unsupported boundary kind: {boundary.kind}")
        normal = torch.tensor(boundary.normal, device=device, dtype=cfg.dtype)
        normal = normal / torch.linalg.norm(normal).clamp_min(cfg.eps)
        boundary_x = _rand_uniform(
            n,
            cfg.domain_radius,
            device=device,
            dtype=cfg.dtype,
            generator=gen,
        )
        boundary_x = boundary_x - (boundary_x @ normal)[:, None] * normal[None, :]
        boundary_x = boundary_x + normal * float(boundary.offset)
        boundary_y = _rand_uniform(
            n,
            cfg.domain_radius,
            device=device,
            dtype=cfg.dtype,
            generator=gen,
        )
        pairs["boundary"] = (boundary_x, boundary_y)

    return pairs


def reciprocity_error(
    eval_fn: EvalFn,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    g_xy = eval_fn(x, y)
    g_yx = eval_fn(y, x)
    denom = torch.max(g_xy.abs().max(), g_yx.abs().max()).clamp_min(eps)
    return (g_xy - g_yx).abs().max() / denom


def far_field_scaling(
    eval_fn: EvalFn,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    g = eval_fn(x, y)
    r = torch.linalg.norm(x - y, dim=1).clamp_min(eps)
    scaled = g * r
    mean = scaled.mean()
    std = scaled.std(unbiased=False)
    cv = std / mean.abs().clamp_min(eps)
    return {"scaled_mean": mean, "scaled_cv": cv}


def boundary_residuals(
    eval_fn: EvalFn,
    boundary: BoundarySpec,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fd_step: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    if boundary.kind != "plane":
        raise ValueError(f"Unsupported boundary kind: {boundary.kind}")

    normal = torch.tensor(boundary.normal, device=x.device, dtype=x.dtype)
    normal = normal / torch.linalg.norm(normal).clamp_min(eps)
    bc_type = boundary.bc_type.lower()

    if bc_type == "dirichlet":
        g = eval_fn(x, y)
        return {"dirichlet_max_abs": g.abs().max()}

    if bc_type == "neumann":
        step = float(fd_step)
        x_plus = x + step * normal
        x_minus = x - step * normal
        g_plus = eval_fn(x_plus, y)
        g_minus = eval_fn(x_minus, y)
        deriv = (g_plus - g_minus) / (2.0 * step)
        return {"neumann_max_abs": deriv.abs().max()}

    raise ValueError(f"Unsupported bc_type: {boundary.bc_type}")


def run_green_checks(
    eval_fn: EvalFn,
    cfg: GreenCheckConfig,
    *,
    boundary: Optional[BoundarySpec] = None,
    oracle_fn: Optional[EvalFn] = None,
) -> Dict[str, float]:
    pairs = sample_region_pairs(cfg, boundary=boundary)
    metrics: Dict[str, float] = {}

    near_x, near_y = pairs["near"]
    metrics["reciprocity_rel_max"] = float(
        reciprocity_error(eval_fn, near_x, near_y, eps=cfg.eps).item()
    )

    far_x, far_y = pairs["far"]
    far_stats = far_field_scaling(eval_fn, far_x, far_y, eps=cfg.eps)
    metrics["far_scaled_mean"] = float(far_stats["scaled_mean"].item())
    metrics["far_scaled_cv"] = float(far_stats["scaled_cv"].item())

    if boundary is not None:
        bnd_x, bnd_y = pairs["boundary"]
        bnd_stats = boundary_residuals(
            eval_fn,
            boundary,
            bnd_x,
            bnd_y,
            fd_step=cfg.fd_step,
            eps=cfg.eps,
        )
        for key, val in bnd_stats.items():
            metrics[key] = float(val.item())

    if oracle_fn is not None:
        for label, (x, y) in pairs.items():
            pred = eval_fn(x, y)
            ref = oracle_fn(x, y)
            diff = pred - ref
            denom = torch.linalg.norm(ref).clamp_min(cfg.eps)
            metrics[f"oracle_rel_l2_{label}"] = float(
                (torch.linalg.norm(diff) / denom).item()
            )

    return metrics
