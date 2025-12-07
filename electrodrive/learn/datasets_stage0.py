from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from electrodrive.utils.config import K_E
from electrodrive.learn.neural_operators import make_unit_sphere_grid


@dataclass
class Stage0SphereRanges:
    """Sampling ranges for Stage-0 grounded sphere with on-axis charge."""

    q: Tuple[float, float] = (0.5, 2.0)
    a: Tuple[float, float] = (0.5, 2.0)
    z0_over_a: Tuple[float, float] = (1.05, 3.0)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _analytic_potential_on_sphere_grid(
    q: float,
    z0: float,
    a: float,
    center: Tuple[float, float, float],
    theta: torch.Tensor,
    phi: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Analytic potential for grounded sphere + on-axis point charge on the surface grid.
    """
    cx, cy, cz = center
    # Points on the sphere surface in world coordinates
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    pts = torch.stack([x, y, z], dim=-1) * float(a)
    pts = pts + torch.tensor([cx, cy, cz], dtype=dtype, device=theta.device)

    # Charge position and image charge
    x0, y0, z0_abs = cx, cy, cz + float(z0)
    r0_vec = torch.tensor([0.0, 0.0, float(z0)], dtype=dtype, device=theta.device)
    r0_norm = torch.linalg.norm(r0_vec).clamp_min(1e-9)
    q_img = -(float(a) / r0_norm) * float(q)
    r_img_vec = (float(a) * float(a) / (r0_norm * r0_norm)) * r0_vec
    r_img_world = torch.tensor([cx, cy, cz], dtype=dtype, device=theta.device) + r_img_vec

    r = torch.linalg.norm(pts - torch.tensor([x0, y0, z0_abs], dtype=dtype, device=theta.device), dim=-1).clamp_min(1e-9)
    ri = torch.linalg.norm(pts - r_img_world, dim=-1).clamp_min(1e-9)
    V = K_E * (float(q) / r + q_img / ri)
    return V.to(dtype=dtype)


class Stage0SphereAxisDataset(Dataset):
    """
    Deterministic Stage-0 dataset for SphereFNO training on on-axis grounded spheres.
    """

    def __init__(
        self,
        n_samples: int,
        *,
        ranges: Stage0SphereRanges = Stage0SphereRanges(),
        n_theta: int = 64,
        n_phi: int = 128,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.ranges = ranges
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.dtype = dtype
        self.seed = int(seed)

        theta, phi = make_unit_sphere_grid(self.n_theta, self.n_phi, dtype=dtype)
        self.registered_theta = theta
        self.registered_phi = phi

    def __len__(self) -> int:
        return self.n_samples

    def _rng_for_idx(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx)

    def _sample_params(self, rng: np.random.Generator) -> Tuple[float, float, float]:
        q = float(rng.uniform(*self.ranges.q))
        a = float(rng.uniform(*self.ranges.a))
        z0_over_a = float(rng.uniform(*self.ranges.z0_over_a))
        z0 = z0_over_a * a * rng.choice([-1.0, 1.0])
        return q, z0, a

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng_for_idx(int(idx))
        q, z0, a = self._sample_params(rng)
        V = _analytic_potential_on_sphere_grid(
            q=q,
            z0=z0,
            a=a,
            center=self.ranges.center,
            theta=self.registered_theta,
            phi=self.registered_phi,
            dtype=self.dtype,
        )
        params = torch.tensor([q, z0, a], dtype=self.dtype)
        return {
            "params": params,
            "V": V,
        }
