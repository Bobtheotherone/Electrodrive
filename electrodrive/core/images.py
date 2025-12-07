# electrodrive/core/images.py
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List

import numpy as np

from electrodrive.utils.config import K_E

Vec3 = Tuple[float, float, float]


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vmul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(max(1e-300, _dot(a, a)))


# 2D helpers for cylinder/line constructions
def _norm2d_sq(a: Tuple[float, float]) -> float:
    return a[0] * a[0] + a[1] * a[1]


def _norm2d(a: Tuple[float, float]) -> float:
    return math.sqrt(_norm2d_sq(a))


@dataclass
class AnalyticSolution:
    """Container for V(x,y,z) and metadata."""
    V: Callable[[float, float, float], float]
    meta: Dict[str, object] = field(default_factory=dict)

    def eval(self, p: Vec3) -> float:
        return float(self.V(*p))


class ImagesSolution:
    # (this is your existing slow, pure-Python implementation)
    # def V(self, x, y, z): ...
    # def eval(self, p): return float(self.V(*p))
    pass

# --- Fast, torch-vectorized MoI evaluator ------------------------------------
from dataclasses import dataclass
import contextlib
import torch
import math

@dataclass
class FastImagesSolution:
    """
    Vectorized evaluator for sum_i q_i / |x - r_i|.
    Keeps charges/positions as device tensors; supports batched queries.
    """
    q: torch.Tensor         # (M,)
    pos: torch.Tensor       # (M, 3)
    device: torch.device
    dtype: torch.dtype

    @staticmethod
    def from_numpy(charges_np, positions_np, device="cuda", dtype="float32"):
        dev = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        dt = getattr(torch, dtype)
        q   = torch.as_tensor(charges_np, device=dev, dtype=dt)
        pos = torch.as_tensor(positions_np, device=dev, dtype=dt)
        return FastImagesSolution(q=q, pos=pos, device=dev, dtype=dt)

    def eval_batch(self, X):
        """
        X: (N,3) np.ndarray or torch.Tensor
        returns: (N,) torch.Tensor (on self.device)
        """
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        # [N,1,3] - [1,M,3] -> [N,M,3]
        diff = X[:, None, :] - self.pos[None, :, :]
        # Use rsqrt for speed; clamp to avoid NaN at coincident points
        inv_r = torch.rsqrt((diff * diff).sum(dim=-1).clamp_min(1e-12))
        V = (self.q * inv_r).sum(dim=1)
        return V

    # Compatibility shim for existing code that calls .eval((x,y,z))
    def eval(self, p):
        out = self.eval_batch([p])
        return float(out[0].detach().cpu())

# Factory: wrap an existing (slow) solution with a fast sibling, if possible.
def as_fast_solution(slow_solution, *, device="cuda", dtype="float32"):
    """
    slow_solution must expose iterable of image charges & positions, e.g.:
        slow_solution.images -> list of (q_i, (x_i,y_i,z_i))
    """
    if not hasattr(slow_solution, "images"):
        return slow_solution  # fallback
    charges = []
    positions = []
    for qi, ri in slow_solution.images:
        charges.append(qi)
        positions.append(ri)
    if len(charges) == 0:
        return slow_solution
    return FastImagesSolution.from_numpy(charges, positions, device=device, dtype=dtype)


# ---------- Plane (half-space, grounded at z=0) ----------

def potential_plane_halfspace(q: float, r0: Vec3) -> AnalyticSolution:
    """
    Potential in z>0 due to q at r0=(x0,y0,z0) and image -q at (x0,y0,-z0).
    Grounded plane at z=0, conductor occupies z<=0.
    """
    x0, y0, z0 = r0
    if z0 <= 0:
        raise ValueError("Point charge must lie in z>0 for plane half-space case.")
    img = (x0, y0, -z0)
    qimg = -q

    def V(x: float, y: float, z: float) -> float:
        r = _norm((x - x0, y - y0, z - z0))
        ri = _norm((x - img[0], y - img[1], z - img[2]))
        return K_E * (q / r + qimg / ri)

    meta = {
        "geometry": "plane",
        "charge": q,
        "r0": r0,
        "image_charge": qimg,
        "image_pos": img,
        "work_formula": "plane_image_halfspace",
    }
    return AnalyticSolution(V, meta)


def work_to_infinity_plane(q: float, d: float) -> float:
    """
    Known result (for reference configuration):
      W = +K_E * q^2 / (16 d)
    """
    if d <= 0:
        raise ValueError("d must be positive for plane case.")
    return K_E * q * q / (16.0 * d)


# ---------- Grounded sphere (Kelvin inversion) ----------

def _scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def potential_sphere_grounded(
    q: float,
    r0: Vec3,
    center: Vec3,
    a: float,
) -> AnalyticSolution:
    """
    Charge q at r0; grounded sphere radius a centered at center.

    Valid for |r0-center| < a (inside) or |r0-center| > a (outside);
    image placed by Kelvin transform:
      q' = -q * a / d
      r'_local = (a^2 / d^2) * (r0 - center)
      r' = center + r'_local
    """
    rc = _vsub(r0, center)
    d = _norm(rc)
    if d == 0.0:
        raise ValueError("Charge at sphere center is singular for inversion.")

    r_img_local = _scale(rc, (a * a) / (d * d))
    q_img = -q * (a / d)
    r_img = _vadd(center, r_img_local)

    def V(x: float, y: float, z: float) -> float:
        r = _norm((x - r0[0], y - r0[1], z - r0[2]))
        ri = _norm((x - r_img[0], y - r_img[1], z - r_img[2]))
        return K_E * (q / r + q_img / ri)

    meta = {
        "geometry": "sphere",
        "charge": q,
        "center": center,
        "radius": a,
        "r0": r0,
        "image_charge": q_img,
        "image_pos": r_img,
    }
    return AnalyticSolution(V, meta)


def force_on_charge_near_grounded_sphere(
    q: float,
    r0: Vec3,
    center: Vec3,
    a: float,
) -> Tuple[float, float, float]:
    """
    Force on real charge q at r0 due to the image charge inside a grounded sphere.

      q_img = -q * a / d
      r_img = center + (a^2 / d^2) * (r0 - center)
      F = Coulomb force from q_img at r_img.
    """
    rc = _vsub(r0, center)
    d = _norm(rc)
    if d <= 0.0:
        return (0.0, 0.0, 0.0)

    r_img_local = _scale(rc, (a * a) / (d * d))
    q_img = -q * (a / d)
    r_img = _vadd(center, r_img_local)

    R = _vsub(r0, r_img)
    dist = _norm(R)
    if dist <= 0.0:
        return (0.0, 0.0, 0.0)

    F_mag = K_E * q * q_img / (dist * dist)
    Fx, Fy, Fz = _vmul(R, F_mag / dist)
    return Fx, Fy, Fz


# ---------- Additional analytic helpers for learning stack ----------

def potential_line_cylinder2d_grounded(
    lambda_c: float,
    a: float,
    r0_2d: Tuple[float, float],
) -> AnalyticSolution:
    """
    Approximate 2D line-charge near grounded cylinder via single image (stub).

    Purpose:
      - Allow electrodrive.learn.dataset to synthesize 'cylinder2D' problems.
      - Not a high-precision reference; suitable for training/eval smoke tests.

    Geometry:
      - Grounded conducting cylinder of radius a at origin.
      - Real line-charge density lambda_c at (r0, 0), r0 > a.
      - Image line-charge at (a^2 / r0, 0) with density lambda_img = -lambda_c * a / r0.
    """
    x0, y0 = float(r0_2d[0]), float(r0_2d[1])
    r0 = math.hypot(x0, y0)
    if r0 <= a:
        raise ValueError("Source for cylinder2D must be outside the cylinder (r0 > a).")

    # Image on same radial line
    r_img = (a * a) / r0
    scale = r_img / r0
    x_img, y_img = x0 * scale, y0 * scale
    lambda_img = -lambda_c * (a / r0)

    def V(x: float, y: float, z: float) -> float:
        # 2D electrostatics: potential ~ ln(r); we use logs as qualitative stand-ins.
        r_real = math.hypot(x - x0, y - y0) + 1e-18
        r_im = math.hypot(x - x_img, y - y_img) + 1e-18
        return float(
            K_E * (lambda_c * math.log(1.0 / r_real) + lambda_img * math.log(1.0 / r_im))
        )

    meta = {
        "geometry": "cylinder2D",
        "lambda": lambda_c,
        "radius": a,
        "r0_2d": (x0, y0),
        "image_lambda": lambda_img,
        "image_pos_2d": (x_img, y_img),
    }
    return AnalyticSolution(V, meta)


def potential_parallel_planes_subset(
    q: float,
    r0: Vec3,
    d: float,
    N_terms: int = 20,
) -> AnalyticSolution:
    """
    Approximate method-of-images potential between two grounded parallel planes at z=±d.

    - Real charge q at r0 = (x0, y0, z0) with |z0| < d.
    - Two grounded planes at z = +d and z = -d.
    - Uses a truncated infinite image lattice: suitable for learning/eval, not P1-grade proofs.
    """
    x0, y0, z0 = map(float, r0)
    if abs(z0) >= d:
        raise ValueError("Charge must lie between the planes (|z0| < d).")

    images: List[Tuple[float, float, float, float]] = []

    # Standard image construction: alternating-sign lattice at
    #   z_n = 2 n d + (-1)^n z0 with charge (-1)^n q.
    for n in range(-N_terms, N_terms + 1):
        sign = -1.0 if (n % 2) else 1.0
        z_n = 2.0 * n * d + sign * z0
        images.append((sign * q, x0, y0, z_n))

    def V(x: float, y: float, z: float) -> float:
        pot = 0.0
        for q_i, xi, yi, zi in images:
            dx = x - xi
            dy = y - yi
            dz = z - zi
            r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-18
            pot += q_i / r
        return float(K_E * pot)

    meta = {
        "geometry": "parallel_planes",
        "q": q,
        "r0": r0,
        "d": d,
        "N_terms": N_terms,
    }
    return AnalyticSolution(V, meta)
