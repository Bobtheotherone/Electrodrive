from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from electrodrive.utils.config import K_E


@dataclass
class ThreeLayerConfig:
    eps1: float
    eps2: float
    eps3: float
    h: float
    q: float
    r0: Tuple[float, float, float]
    n_k: int = 256
    k_max: float | None = None


def _reflection_coefficients(eps1: float, eps2: float, eps3: float, k: torch.Tensor) -> torch.Tensor:
    """
    Static TM reflection coefficient for a three-layer stack seen from region 1.

    R12 = (eps1 - eps2)/(eps1 + eps2)
    R21 = (eps2 - eps1)/(eps1 + eps2) = -R12
    R23 = (eps2 - eps3)/(eps2 + eps3)
    R_eff = R12 + (T12*T21*R23*exp(-2 k h)) / (1 - R21*R23*exp(-2 k h))
    """
    R12 = (eps1 - eps2) / (eps1 + eps2)
    R21 = -R12
    R23 = (eps2 - eps3) / (eps2 + eps3)
    T12 = 2.0 * eps2 / (eps1 + eps2)
    T21 = 2.0 * eps1 / (eps1 + eps2)
    exp_term = torch.exp(-2.0 * k * torch.tensor(1.0, device=k.device, dtype=k.dtype))
    # The exp_term is scaled later by h in the caller; keep base form here.
    return R12, R21, R23, T12 * T21


def potential_three_layer_region1(
    points: torch.Tensor,
    cfg: ThreeLayerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Evaluate the potential in region 1 (z >= 0) for a three-layer planar stack:
      region1: z > 0, permittivity eps1 (source resides here)
      region2: -h < z < 0, permittivity eps2
      region3: z < -h, permittivity eps3

    Uses a Sommerfeld integral with an effective reflection coefficient
    capturing multiple reflections in the slab. This is a numerical
    reference (not an image approximation).
    """
    points = points.to(device=device, dtype=dtype)
    x0, y0, z0 = map(float, cfg.r0)
    q = float(cfg.q)
    eps1, eps2, eps3 = float(cfg.eps1), float(cfg.eps2), float(cfg.eps3)
    h = float(cfg.h)

    rho = torch.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2).clamp_min(1e-9)
    z = points[:, 2]

    r_direct = torch.sqrt(rho * rho + (z - z0) ** 2).clamp_min(1e-9)
    direct = K_E * q / (4.0 * math.pi * eps1) * (1.0 / r_direct)

    # Quadrature grid
    n_k = max(64, int(cfg.n_k))
    if cfg.k_max is not None and cfg.k_max > 0:
        k_max = float(cfg.k_max)
    else:
        length_scale = max(1e-3, min(z0 + h, z0 + abs(float(points[:, 2].max().item())) + h))
        k_max = 40.0 / length_scale
    k = torch.linspace(1e-5, k_max, n_k, device=device, dtype=dtype)
    dk = k[1] - k[0]

    R12 = (eps1 - eps2) / (eps1 + eps2)
    R21 = -R12
    R23 = (eps2 - eps3) / (eps2 + eps3)
    T12 = 2.0 * eps2 / (eps1 + eps2)
    T21 = 2.0 * eps1 / (eps1 + eps2)
    exp_h = torch.exp(-2.0 * k * h)
    R_eff = R12 + (T12 * T21 * R23 * exp_h) / (1.0 - R21 * R23 * exp_h + 1e-12)

    # Batch evaluate Bessel and exponentials.
    kr = torch.outer(k, rho)  # [n_k, N]
    j0 = torch.special.bessel_j0(kr)

    exp_dir = torch.exp(-torch.outer(k, torch.abs(z - z0)))
    exp_ref = torch.exp(-torch.outer(k, z + z0)) * R_eff[:, None]
    integrand = k[:, None] * (exp_dir + exp_ref) * j0  # [n_k, N]

    integral = torch.sum(integrand, dim=0) * dk
    reflected = (q / (2.0 * math.pi * 2.0 * eps1)) * integral
    return direct + reflected


def make_three_layer_solution(cfg: ThreeLayerConfig):
    """Return a lightweight AnalyticSolution-like object."""

    def _eval(p):
        pts = torch.tensor([p], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
        return float(potential_three_layer_region1(pts, cfg, pts.device, pts.dtype)[0].item())

    from electrodrive.core.images import AnalyticSolution

    meta: Dict[str, object] = {
        "kind": "planar_three_layer",
        "eps1": cfg.eps1,
        "eps2": cfg.eps2,
        "eps3": cfg.eps3,
        "h": cfg.h,
        "q": cfg.q,
        "r0": cfg.r0,
        "n_k": cfg.n_k,
        "k_max": cfg.k_max,
    }
    return AnalyticSolution(V=_eval, meta=meta)
