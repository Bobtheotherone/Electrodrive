from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch

from .gates import gateB_bc, gateC_asymptotics, gateD_stability


def _as_spec_dict(spec: object) -> dict:
    if isinstance(spec, dict):
        return spec
    if hasattr(spec, "to_json"):
        return spec.to_json()  # type: ignore[no-any-return]
    raise TypeError("spec must be dict-like or CanonicalSpec")


def _eval_tensor(fn: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    out = fn(pts)
    if isinstance(out, tuple):
        out = out[0]
    if not torch.is_tensor(out):
        raise ValueError("candidate_eval must return a torch.Tensor")
    return out.flatten()


def proxy_gateB(
    spec: object,
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_xy: int,
    delta: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> Dict[str, float]:
    spec_dict = _as_spec_dict(spec)
    interfaces = gateB_bc._interfaces(list(spec_dict.get("dielectrics", []) or []))
    if not interfaces or n_xy <= 0:
        return {"proxy_gateB_max_v_jump": 0.0, "proxy_gateB_max_d_jump": 0.0}

    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    xy = engine.draw(n_xy).to(device=device, dtype=dtype)
    xy = (xy - 0.5) * 2.0

    max_v_jump = 0.0
    max_d_jump = 0.0
    for z_val, eps_up, eps_down in interfaces:
        pts_up = torch.stack(
            [xy[:, 0], xy[:, 1], torch.full((n_xy,), z_val + delta, device=device, dtype=dtype)],
            dim=1,
        )
        pts_dn = torch.stack(
            [xy[:, 0], xy[:, 1], torch.full((n_xy,), z_val - delta, device=device, dtype=dtype)],
            dim=1,
        )
        pts_up = pts_up.detach().clone().requires_grad_(True)
        pts_dn = pts_dn.detach().clone().requires_grad_(True)
        V_up = _eval_tensor(candidate_eval, pts_up)
        V_dn = _eval_tensor(candidate_eval, pts_dn)
        grad_up = torch.autograd.grad(V_up, pts_up, grad_outputs=torch.ones_like(V_up), create_graph=False)[0]
        grad_dn = torch.autograd.grad(V_dn, pts_dn, grad_outputs=torch.ones_like(V_dn), create_graph=False)[0]
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        d_jump = torch.abs(eps_up * torch.sum(grad_up * normal, dim=1) - eps_down * torch.sum(grad_dn * normal, dim=1))
        v_jump = torch.abs(V_up - V_dn)
        max_v_jump = max(max_v_jump, float(torch.max(v_jump).item()))
        max_d_jump = max(max_d_jump, float(torch.max(d_jump).item()))
    return {
        "proxy_gateB_max_v_jump": max_v_jump,
        "proxy_gateB_max_d_jump": max_d_jump,
    }


def _as_pair(radii: Iterable[float]) -> Tuple[float, float]:
    vals = list(radii)
    if len(vals) != 2:
        raise ValueError("radii must be length-2 iterable")
    return float(vals[0]), float(vals[1])


def proxy_gateC(
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    *,
    near_radii: Iterable[float],
    far_radii: Iterable[float],
    n_dir: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> Dict[str, float]:
    near_min, near_max = _as_pair(near_radii)
    far_min, far_max = _as_pair(far_radii)

    far_pts, far_r = gateC_asymptotics._sample_radial(n_dir, far_min, far_max, device, dtype, seed=seed)
    near_pts, near_r = gateC_asymptotics._sample_radial(n_dir, near_min, near_max, device, dtype, seed=seed + 1)

    far_vals = torch.abs(_eval_tensor(candidate_eval, far_pts))
    near_vals = torch.abs(_eval_tensor(candidate_eval, near_pts))
    far_slope = gateC_asymptotics._fit_slope(far_r, far_vals)
    near_slope = gateC_asymptotics._fit_slope(near_r, near_vals)

    spurious_tol = 1e3
    spurious_fraction = float(torch.mean((far_vals > spurious_tol).float()).item())
    return {
        "proxy_gateC_far_slope": float(far_slope),
        "proxy_gateC_near_slope": float(near_slope),
        "proxy_gateC_spurious_fraction": float(spurious_fraction),
    }


def proxy_gateD(
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    pts: torch.Tensor,
    *,
    delta: float,
    seed: int = 0,
) -> Dict[str, float]:
    gen = torch.Generator(device=pts.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(pts.shape, device=pts.device, dtype=pts.dtype, generator=gen)
    perturb = noise * delta
    base_val = _eval_tensor(candidate_eval, pts)
    pert_val = _eval_tensor(candidate_eval, pts + perturb)
    diff = base_val - pert_val
    denom = torch.norm(base_val).clamp_min(1e-8)
    rel_change = float((torch.norm(diff) / denom).item())
    var_base = float(torch.var(base_val).item())
    return {
        "proxy_gateD_rel_change": rel_change,
        "proxy_gateD_variance": var_base,
    }


__all__ = ["proxy_gateB", "proxy_gateC", "proxy_gateD"]
