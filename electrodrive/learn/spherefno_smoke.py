from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from electrodrive.core.images import potential_sphere_grounded
from electrodrive.learn.neural_operators import (
    SphereFNOSurrogate,
    extract_stage0_sphere_params,
)
from electrodrive.orchestration.parser import CanonicalSpec


def _sample_points(a: float, center: Tuple[float, float, float], n: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """Sample random points outside the sphere for a quick smoke test."""
    cx, cy, cz = center
    dirs = torch.from_numpy(rng.standard_normal((n, 3))).to(device=device, dtype=torch.float32)
    dirs = dirs / (torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(1e-9))
    radii = torch.from_numpy(rng.uniform(1.05 * a, 3.0 * a, size=n)).to(device=device, dtype=torch.float32)
    pts = dirs * radii.unsqueeze(1)
    pts += torch.tensor([cx, cy, cz], device=device, dtype=torch.float32)
    return pts


def _analytic_targets(q: float, z0: float, a: float, center, pts: torch.Tensor) -> torch.Tensor:
    sol = potential_sphere_grounded(q, (center[0], center[1], center[2] + z0), center, a)
    vals = [sol.eval(tuple(p.cpu().tolist())) for p in pts]
    return torch.tensor(vals, device=pts.device, dtype=pts.dtype)


def run_smoke(spec_path: Path, ckpt_path: Path, device: torch.device, n_samples: int = 8) -> int:
    spec = CanonicalSpec.from_json(json.loads(spec_path.read_text()))
    params = extract_stage0_sphere_params(spec)
    if params is None:
        print("Spec is not Stage-0 on-axis sphere; aborting.")
        return 1

    surrogate = SphereFNOSurrogate.from_checkpoint(str(ckpt_path), device=device)
    if not surrogate.is_ready():
        print("Surrogate failed validation or is not ready.")
        return 1

    q, z0, a, center = params
    rng = np.random.default_rng(0)
    pts = _sample_points(a, center, n_samples, rng, device)
    pred = surrogate.evaluate_points((q, z0, a), pts, center=center)
    target = _analytic_targets(q, z0, a, center, pts)

    diff = pred - target
    rel_l2 = float(torch.linalg.norm(diff) / torch.linalg.norm(target).clamp_min(1e-12))
    rel_linf = float(torch.max(torch.abs(diff)) / torch.max(torch.abs(target)).clamp_min(1e-12))

    print(json.dumps({
        "ckpt": str(ckpt_path),
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "validated": surrogate.validated,
    }, indent=2))
    return 0
