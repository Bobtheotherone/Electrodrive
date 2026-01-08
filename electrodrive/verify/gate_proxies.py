from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Tuple

import torch

from .gates import gateA_pde, gateB_bc, gateC_asymptotics, gateD_stability


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


def _laplacian_eval_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return dtype


def _dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    return str(dtype)


def _bounds_from_spec(spec: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    try:
        return gateA_pde._bounds_from_spec(spec)  # type: ignore[attr-defined]
    except Exception:
        charges = spec.get("charges", []) or []
        if charges:
            pts = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], dtype=torch.float32)
            lo = torch.min(pts, dim=0).values - 1.0
            hi = torch.max(pts, dim=0).values + 1.0
            return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)


def _interface_planes_from_spec(spec: Dict[str, Any]) -> list[float]:
    try:
        return gateA_pde._interface_planes_from_spec(spec)  # type: ignore[attr-defined]
    except Exception:
        planes = []
        for layer in spec.get("dielectrics", []) or []:
            z_min = layer.get("z_min", None)
            z_max = layer.get("z_max", None)
            if z_min is not None:
                planes.append(float(z_min))
            if z_max is not None:
                planes.append(float(z_max))
        if not planes:
            return []
        uniq = sorted({z for z in planes if math.isfinite(z)})
        return uniq


def _filter_interface_band(pts: torch.Tensor, planes: list[float], band: float) -> torch.Tensor:
    try:
        return gateA_pde._filter_interface_band(pts, planes, band)  # type: ignore[attr-defined]
    except Exception:
        if band <= 0.0 or not planes or pts.numel() == 0:
            return pts
        z = pts[:, 2]
        plane_tensor = torch.tensor(planes, device=pts.device, dtype=pts.dtype)
        dists = torch.abs(z[:, None] - plane_tensor[None, :])
        mask = torch.all(dists >= band, dim=1)
        return pts[mask]


def _sample_interior_fallback(
    spec: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    *,
    seed: int,
    exclusion_radius: float,
    interface_planes: list[float] | None = None,
    interface_band: float = 0.0,
    extra_radius: float = 0.0,
    max_attempts: int = 8,
) -> torch.Tensor:
    (x0, x1), (y0, y1), (z0, z1) = _bounds_from_spec(spec)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    span = torch.tensor([x1 - x0, y1 - y0, z1 - z0], device=device, dtype=dtype)
    base = torch.tensor([x0, y0, z0], device=device, dtype=dtype)
    interface_planes = interface_planes or []
    min_dist = exclusion_radius + extra_radius
    charges = spec.get("charges", []) or []
    charge_pos = None
    if charges:
        charge_pos = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], device=device, dtype=dtype)

    collected: list[torch.Tensor] = []
    total = 0
    attempts = 0
    while total < n and attempts < max_attempts:
        remaining = n - total
        draw_n = max(remaining * 2, 32)
        pts = engine.draw(draw_n).to(device=device, dtype=dtype)
        pts = pts * span + base
        if charge_pos is not None and min_dist > 0.0:
            dists = torch.cdist(pts, charge_pos)
            mask = torch.all(dists > min_dist, dim=1)
            pts = pts[mask]
        pts = _filter_interface_band(pts, interface_planes, interface_band)
        if pts.numel() > 0:
            collected.append(pts)
            total += int(pts.shape[0])
        attempts += 1

    if collected:
        pts = torch.cat(collected, dim=0)[:n].contiguous()
    else:
        pts = torch.zeros(0, 3, device=device, dtype=dtype)
    return pts.contiguous()


def _sample_interior_gateA(
    spec: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    *,
    seed: int,
    exclusion_radius: float,
    interface_planes: list[float] | None = None,
    interface_band: float = 0.0,
    extra_radius: float = 0.0,
    max_attempts: int = 8,
) -> torch.Tensor:
    if device.type == "cuda":
        try:
            return gateA_pde._sample_interior(  # type: ignore[attr-defined]
                spec,
                device,
                dtype,
                n,
                seed=seed,
                exclusion_radius=exclusion_radius,
                interface_planes=interface_planes,
                interface_band=interface_band,
                extra_radius=extra_radius,
                max_attempts=max_attempts,
            )
        except Exception:
            pass
    return _sample_interior_fallback(
        spec,
        device,
        dtype,
        n,
        seed=seed,
        exclusion_radius=exclusion_radius,
        interface_planes=interface_planes,
        interface_band=interface_band,
        extra_radius=extra_radius,
        max_attempts=max_attempts,
    )


def _laplacian_autograd(candidate_eval: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    pts = pts.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        V = _eval_tensor(candidate_eval, pts)
    if V.shape[0] != pts.shape[0]:
        raise ValueError("candidate_eval must return one value per point")
    lap = torch.zeros_like(V)
    for idx in range(pts.shape[0]):
        grad = torch.autograd.grad(
            V[idx],
            pts,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            continue
        grad_i = grad[idx]
        second = 0.0
        for dim in range(3):
            grad2 = torch.autograd.grad(
                grad_i[dim],
                pts,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad2 is None:
                continue
            second = second + grad2[idx, dim]
        lap[idx] = second
    return lap.detach()


def _laplacian_finite_diff(
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    pts: torch.Tensor,
    *,
    h: float,
) -> torch.Tensor:
    V0 = _eval_tensor(candidate_eval, pts)
    lap = torch.zeros_like(V0)
    eye = torch.eye(3, device=pts.device, dtype=pts.dtype) * h
    for dim in range(3):
        offset = eye[dim].unsqueeze(0)
        plus = _eval_tensor(candidate_eval, pts + offset)
        minus = _eval_tensor(candidate_eval, pts - offset)
        lap = lap + (plus - 2.0 * V0 + minus) / (h * h)
    return lap.detach()


def proxy_gateA(
    spec: object,
    candidate_eval: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_interior: int,
    exclusion_radius: float,
    fd_h: float,
    prefer_autograd: bool,
    interface_band: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
    linf_tol: float = 5e-3,
    autograd_max_samples: int | None = None,
    fd_max_samples: int = 128,
    fd_stencil_margin: float = 1.0,
    resample_max_attempts: int = 8,
) -> Dict[str, Any]:
    spec_dict = _as_spec_dict(spec)
    lap_dtype = _laplacian_eval_dtype(dtype)
    eval_dtype_label = _dtype_label(lap_dtype)
    if autograd_max_samples is None:
        autograd_max_samples = n_interior
    method = "none"
    n_used = 0
    fail_value = 1e12
    if n_interior <= 0:
        return {
            "proxy_gateA_linf": fail_value,
            "proxy_gateA_l2": fail_value,
            "proxy_gateA_p95": fail_value,
            "proxy_gateA_status": "fail",
            "proxy_gateA_worst_ratio": fail_value,
            "proxy_gateA_method": method,
            "proxy_gateA_n_used": n_used,
            "proxy_gateA_eval_dtype": eval_dtype_label,
        }

    interface_planes = _interface_planes_from_spec(spec_dict)
    pts = _sample_interior_gateA(
        spec_dict,
        device,
        lap_dtype,
        n_interior,
        seed=seed,
        exclusion_radius=exclusion_radius,
        interface_planes=interface_planes,
        interface_band=interface_band,
        max_attempts=resample_max_attempts,
    )
    if pts.shape[0] < n_interior:
        return {
            "proxy_gateA_linf": fail_value,
            "proxy_gateA_l2": fail_value,
            "proxy_gateA_p95": fail_value,
            "proxy_gateA_status": "fail",
            "proxy_gateA_worst_ratio": fail_value,
            "proxy_gateA_method": method,
            "proxy_gateA_n_used": n_used,
            "proxy_gateA_eval_dtype": eval_dtype_label,
        }

    lap = torch.zeros(0, device=pts.device, dtype=pts.dtype)
    if prefer_autograd and pts.shape[0] <= autograd_max_samples:
        try:
            lap = _laplacian_autograd(candidate_eval, pts)
            method = "autograd"
            n_used = int(pts.shape[0])
            finite_mask = torch.isfinite(lap)
            nonfinite_frac = 1.0 - float(finite_mask.float().mean().item())
            if nonfinite_frac > 0.1:
                lap = torch.zeros(0, device=pts.device, dtype=pts.dtype)
                method = "none"
                n_used = 0
        except Exception:
            lap = torch.zeros(0, device=pts.device, dtype=pts.dtype)
            method = "none"
            n_used = 0

    if lap.numel() == 0:
        fd_limit = max(16, min(int(fd_max_samples), n_interior))
        fd_pts = _sample_interior_gateA(
            spec_dict,
            device,
            lap_dtype,
            fd_limit,
            seed=seed + 1,
            exclusion_radius=exclusion_radius,
            interface_planes=interface_planes,
            interface_band=interface_band + float(fd_h),
            extra_radius=float(fd_h) * fd_stencil_margin,
            max_attempts=resample_max_attempts,
        )
        if fd_pts.shape[0] < fd_limit:
            return {
                "proxy_gateA_linf": fail_value,
                "proxy_gateA_l2": fail_value,
                "proxy_gateA_p95": fail_value,
                "proxy_gateA_status": "fail",
                "proxy_gateA_worst_ratio": fail_value,
                "proxy_gateA_method": method,
                "proxy_gateA_n_used": n_used,
                "proxy_gateA_eval_dtype": eval_dtype_label,
            }
        lap = _laplacian_finite_diff(candidate_eval, fd_pts, h=float(fd_h))
        pts = fd_pts
        method = "finite_diff"
        n_used = int(fd_pts.shape[0])

    finite_mask = torch.isfinite(lap)
    if not torch.any(finite_mask):
        linf = fail_value
        l2 = fail_value
        p95 = fail_value
    else:
        lap = lap[finite_mask]
        abs_lap = torch.abs(lap)
        linf = float(torch.max(abs_lap).item()) if abs_lap.numel() > 0 else fail_value
        abs_lap_f64 = abs_lap.double()
        l2 = float(torch.sqrt(torch.mean(abs_lap_f64 * abs_lap_f64)).item()) if abs_lap_f64.numel() > 0 else fail_value
        p95 = float(torch.quantile(abs_lap_f64, 0.95).item()) if abs_lap_f64.numel() > 0 else fail_value

    worst_ratio = float(max(linf / linf_tol, l2 / linf_tol)) if linf_tol > 0 else float("inf")
    status = "fail"
    if linf <= linf_tol and l2 <= linf_tol:
        status = "pass"
    elif linf <= 2.0 * linf_tol:
        status = "borderline"
    return {
        "proxy_gateA_linf": linf,
        "proxy_gateA_l2": l2,
        "proxy_gateA_p95": p95,
        "proxy_gateA_status": status,
        "proxy_gateA_worst_ratio": worst_ratio,
        "proxy_gateA_method": method,
        "proxy_gateA_n_used": n_used,
        "proxy_gateA_eval_dtype": eval_dtype_label,
    }


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


__all__ = ["proxy_gateA", "proxy_gateB", "proxy_gateC", "proxy_gateD"]
