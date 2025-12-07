# electrodrive/core/pinn_data.py
"""
Minimal, robust PINN data samplers used by electrodrive.core.pinn and the
learning stack.

Design goals:
- Torch-only (no NumPy at call sites).
- Safe on CPU-only machines.
- Small, stable API surface.

API
---

InteriorSampler:
    InteriorSampler(domain: list[list[float]], seed: int = 42)
        .sample(n: int, device: torch.device, dtype: torch.dtype,
                avoid_points: Optional[list[tuple[float,float,float]]] = None,
                avoid_radius: float = 1e-3, **kwargs)
        -> torch.Tensor[n,3]

    - Samples uniformly inside an axis-aligned box.
    - Optionally avoids small balls around given points via rejection resampling.

BoundarySampler:
    BoundarySampler(plane_L: float = 2.0,
                    sphere_radius: Optional[float] = None,
                    sphere_center: Optional[tuple[float,float,float]] = None,
                    seed: int = 42)
        .sample(n: int, device: torch.device, dtype: torch.dtype, **kwargs)
        -> torch.Tensor[n,3]

    - If sphere_radius is set, samples points on that sphere via stratified theta/phi.
    - Else samples points on the patch [-L,L] x [-L,L] at z=0.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import math

import torch


class InteriorSampler:
    """
    Uniform sampler inside an axis-aligned bounding box.

    Parameters
    ----------
    domain: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        Bounds of the sampling region.
    seed: int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        domain: List[List[float]],
        seed: int = 42,
        **_kwargs,
    ) -> None:
        if (
            not isinstance(domain, (list, tuple))
            or len(domain) != 3
            or any(len(b) != 2 for b in domain)
        ):
            raise ValueError(
                "InteriorSampler domain must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."
            )
        self.xmin, self.xmax = float(domain[0][0]), float(domain[0][1])
        self.ymin, self.ymax = float(domain[1][0]), float(domain[1][1])
        self.zmin, self.zmax = float(domain[2][0]), float(domain[2][1])

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))

    def _uniform_box(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        g = self._rng
        x = torch.empty(n, device=device, dtype=dtype).uniform_(
            self.xmin, self.xmax, generator=g
        )
        y = torch.empty(n, device=device, dtype=dtype).uniform_(
            self.ymin, self.ymax, generator=g
        )
        z = torch.empty(n, device=device, dtype=dtype).uniform_(
            self.zmin, self.zmax, generator=g
        )
        return torch.stack((x, y, z), dim=-1)

    def sample(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        avoid_points: Optional[List[Tuple[float, float, float]]] = None,
        avoid_radius: float = 1e-3,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Sample n points inside the box on the given device/dtype.

        If avoid_points is provided, any point within avoid_radius of those
        locations is resampled with a small retry budget.

        This method never raises on rejection failure; it will leave a small
        number of near-avoid points if necessary rather than looping forever.
        """
        if n <= 0:
            return torch.empty(0, 3, device=device, dtype=dtype)

        pts = self._uniform_box(n, device, dtype)
        if not avoid_points:
            return pts

        avoid = torch.tensor(avoid_points, device=device, dtype=dtype)
        if avoid.numel() == 0:
            return pts

        r2 = float(avoid_radius) ** 2
        max_retries = 4  # NOTE: small, fixed budget to avoid unbounded loops

        for _ in range(max_retries):
            # [n, m, 3]
            delta = pts[:, None, :] - avoid[None, :, :]
            dist2 = torch.sum(delta * delta, dim=-1)
            bad = (dist2 < r2).any(dim=1)
            if not bad.any():
                break
            n_bad = int(bad.sum().item())
            pts[bad] = self._uniform_box(n_bad, device, dtype)

        return pts


class BoundarySampler:
    """
    Boundary sampler for simple PINN problems.

    Modes:
    - Plane z=0 over [-L, L] x [-L, L] when sphere_radius is None.
    - Sphere of radius `sphere_radius` centered at `sphere_center` otherwise.
    """

    def __init__(
        self,
        plane_L: float = 2.0,
        sphere_radius: Optional[float] = None,
        sphere_center: Optional[Tuple[float, float, float]] = None,
        seed: int = 42,
        **_kwargs,
    ) -> None:
        self.L = float(plane_L)
        self.sphere_radius = float(sphere_radius) if sphere_radius is not None else None

        if sphere_center is None:
            self.sphere_center = (0.0, 0.0, 0.0)
        else:
            self.sphere_center = (
                float(sphere_center[0]),
                float(sphere_center[1]),
                float(sphere_center[2]),
            )

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))

    def _sample_plane(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        g = self._rng
        L = self.L
        x = torch.empty(n, device=device, dtype=dtype).uniform_(-L, L, generator=g)
        y = torch.empty(n, device=device, dtype=dtype).uniform_(-L, L, generator=g)
        z = torch.zeros(n, device=device, dtype=dtype)
        return torch.stack((x, y, z), dim=-1)

    def _sample_sphere(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Sample points on a sphere using stratified (theta, phi) with mild jitter.
        """
        g = self._rng
        a = float(self.sphere_radius)
        cx, cy, cz = self.sphere_center

        if n <= 0:
            return torch.empty(0, 3, device=device, dtype=dtype)

        m = max(1, int(math.sqrt(max(1, n))))
        k = max(1, (n + m - 1) // m)
        pts = torch.empty(n, 3, device=device, dtype=dtype)
        idx = 0
        for i in range(m):
            for j in range(k):
                if idx >= n:
                    break
                # Base stratified coordinates in [0,1]
                u = (i + 0.5) / m
                v = (j + 0.5) / k
                # Add small jitter for decorrelation
                du = (torch.rand(1, generator=g).item() - 0.5) / max(m, 1)
                dv = (torch.rand(1, generator=g).item() - 0.5) / max(k, 1)
                u_ = min(max(u + du, 0.0), 1.0)
                v_ = (v + dv) % 1.0
                theta = math.acos(1.0 - 2.0 * u_)
                phi = 2.0 * math.pi * v_
                x = cx + a * math.sin(theta) * math.cos(phi)
                y = cy + a * math.sin(theta) * math.sin(phi)
                z = cz + a * math.cos(theta)
                pts[idx, 0] = x
                pts[idx, 1] = y
                pts[idx, 2] = z
                idx += 1
            if idx >= n:
                break

        if idx < n:
            # If grid didn't fill exactly, repeat last valid point (harmless).
            pts[idx:] = pts[idx - 1].clone()

        return pts

    def sample(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Sample n boundary points and return a [n,3] tensor.

        - Behavior is determined at construction time by sphere_radius:
          * If set and positive: sphere mode.
          * Else: plane z=0 patch mode.
        """
        if n <= 0:
            return torch.empty(0, 3, device=device, dtype=dtype)

        if self.sphere_radius is not None and self.sphere_radius > 0.0:
            return self._sample_sphere(n, device, dtype)
        return self._sample_plane(n, device, dtype)