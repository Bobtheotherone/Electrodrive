from __future__ import annotations

import torch
from typing import Callable, Dict, List, Optional, Tuple

from . import _assert_cuda_inputs
from . import GateResult
from ..oracle_types import OracleQuery, OracleResult
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.learn.collocation import _infer_geom_type_from_spec


def _plane_mask(points: torch.Tensor, z: float, tol: float) -> torch.Tensor:
    return torch.isclose(points[:, 2], torch.tensor(z, device=points.device, dtype=points.dtype), atol=tol)


def _sphere_mask(points: torch.Tensor, center: torch.Tensor, radius: float, tol: float) -> torch.Tensor:
    r = torch.linalg.norm(points - center, dim=1)
    return torch.isclose(r, torch.tensor(radius, device=points.device, dtype=points.dtype), atol=tol)


def _sample_boundary(
    spec: CanonicalSpec, device: torch.device, dtype: torch.dtype, n: int, *, seed: int = 0
) -> Optional[torch.Tensor]:
    conductors = getattr(spec, "conductors", None) or []
    if not conductors:
        return None
    torch.manual_seed(seed)
    c = conductors[0]
    ctype = c.get("type")
    if ctype == "plane":
        z = float(c.get("z", 0.0))
        # SobolEngine draws on CPU; move to CUDA immediately to keep sampling GPU-resident.
        xy = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed).draw(n).to(device=device, dtype=dtype)
        xy = (xy - 0.5) * 4.0
        pts = torch.stack([xy[:, 0], xy[:, 1], torch.full((n,), z, device=device, dtype=dtype)], dim=1)
        if not pts.is_cuda:
            raise ValueError("Gate B boundary samples must be CUDA")
        return pts
    if ctype == "sphere":
        radius = float(c.get("radius", 1.0))
        center = torch.tensor(c.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
        theta = torch.rand(n, device=device, dtype=dtype) * 2.0 * torch.pi
        phi = torch.rand(n, device=device, dtype=dtype) * torch.pi
        x = center[0] + radius * torch.sin(phi) * torch.cos(theta)
        y = center[1] + radius * torch.sin(phi) * torch.sin(theta)
        z = center[2] + radius * torch.cos(phi)
        pts = torch.stack([x, y, z], dim=1)
        if not pts.is_cuda:
            raise ValueError("Gate B boundary samples must be CUDA")
        return pts
    return None


def _interfaces(dielectrics: list[dict[str, object]]) -> List[Tuple[float, float, float]]:
    interfaces: List[Tuple[float, float, float]] = []
    for d in dielectrics:
        if "z_max" not in d:
            continue
        z_top = float(d["z_max"])
        eps_top = float(d.get("epsilon", d.get("eps", 1.0)))
        for other in dielectrics:
            if "z_min" not in other:
                continue
            z_bottom = float(other["z_min"])
            if abs(z_bottom - z_top) < 1e-6:
                eps_bottom = float(other.get("epsilon", other.get("eps", 1.0)))
                interfaces.append((z_top, eps_top, eps_bottom))
                break
    return interfaces


def run_gate(
    query: OracleQuery,
    result: OracleResult,
    *,
    config: Optional[Dict[str, object]] = None,
) -> GateResult:
    _assert_cuda_inputs(query, result)
    cfg = dict(config or {})
    eval_fn = cfg.pop("eval_fn", None)
    candidate_eval: Callable[[torch.Tensor], torch.Tensor]
    if not callable(eval_fn):
        raise ValueError("Gate B requires 'eval_fn' callable in config")
    else:
        candidate_eval = eval_fn  # type: ignore[assignment]

    dirichlet_tol = float(cfg.get("tolerance", 1e-3))
    continuity_tol = float(cfg.get("continuity_tol", 5e-3))
    seed = int(cfg.get("seed", 0))
    n_samples = int(cfg.get("n_samples", 96))
    delta = float(cfg.get("interface_delta", 5e-3))

    def _ensure_cuda(t: torch.Tensor) -> torch.Tensor:
        if not t.is_cuda:
            raise ValueError("Gate B expects CUDA tensors from eval_fn")
        return t

    def _grad(points: torch.Tensor) -> torch.Tensor:
        pts = points.detach().clone().requires_grad_(True)
        V = _ensure_cuda(candidate_eval(pts))
        grad = torch.autograd.grad(V, pts, grad_outputs=torch.ones_like(V), create_graph=False, retain_graph=False)[0]
        return grad.detach()

    try:
        spec = CanonicalSpec.from_json(query.spec)
    except Exception:
        patched = {"domain": "auto"}
        patched.update(query.spec)
        spec = CanonicalSpec.from_json(patched)
    geom = _infer_geom_type_from_spec(spec)
    device = query.points.device
    dtype = query.points.dtype
    target_potential = 0.0
    if spec.conductors:
        target_potential = float(spec.conductors[0].get("potential", 0.0))

    has_conductors = bool(spec.conductors)
    points = query.points
    tol = dirichlet_tol
    mask = torch.zeros(points.shape[0], device=device, dtype=torch.bool)
    sampled = False
    V_vals = torch.empty(0, device=device, dtype=dtype)
    max_err = 0.0
    notes: List[str] = []
    if has_conductors:
        if geom == "plane":
            z = float(spec.conductors[0].get("z", 0.0))
            mask = _plane_mask(points, z, tol)
        elif geom == "sphere":
            center = torch.tensor(spec.conductors[0].get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
            radius = float(spec.conductors[0].get("radius", 1.0))
            mask = _sphere_mask(points, center, radius, dirichlet_tol)

        if not torch.any(mask):
            sampled_pts = _sample_boundary(spec, device, dtype, n_samples, seed=seed)
            if sampled_pts is not None:
                sampled = True
                points = sampled_pts
                V_vals = _ensure_cuda(candidate_eval(points))
            else:
                V_vals = result.V if result.V is not None else torch.empty(0, device=device, dtype=dtype)
        else:
            V_vals = result.V[mask] if result.V is not None else torch.empty(0, device=device, dtype=dtype)

        if V_vals.numel() == 0:
            notes.append("no_boundary_samples")
        else:
            max_err = float(torch.max(torch.abs(V_vals - target_potential)).item())
    else:
        points = torch.empty(0, 3, device=device, dtype=dtype)
        notes.append("no_conductors_dirichlet_skipped")
    interface_metrics: List[float] = []
    interface_d_metrics: List[float] = []
    evidence: Dict[str, str] = {}

    dielectrics = getattr(spec, "dielectrics", None) or []
    if dielectrics:
        interfaces = _interfaces(dielectrics)
        if interfaces:
            # SobolEngine draws on CPU; move to CUDA immediately to keep sampling GPU-resident.
            xy = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed + 1).draw(n_samples).to(device=device, dtype=dtype)
            xy = (xy - 0.5) * 2.0
            for z_val, eps_up, eps_down in interfaces:
                pts_upper = torch.stack([xy[:, 0], xy[:, 1], torch.full((n_samples,), z_val + delta, device=device, dtype=dtype)], dim=1)
                pts_lower = torch.stack([xy[:, 0], xy[:, 1], torch.full((n_samples,), z_val - delta, device=device, dtype=dtype)], dim=1)
                V_up = _ensure_cuda(candidate_eval(pts_upper))
                V_low = _ensure_cuda(candidate_eval(pts_lower))
                grad_up = _grad(pts_upper)
                grad_low = _grad(pts_lower)
                normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
                d_jump = torch.abs(eps_up * torch.sum(grad_up * normal, dim=1) - eps_down * torch.sum(grad_low * normal, dim=1))
                v_jump = torch.abs(V_up - V_low)
                interface_metrics.append(float(torch.max(v_jump).item()))
                interface_d_metrics.append(float(torch.max(d_jump).item()))

    max_v_jump = max(interface_metrics) if interface_metrics else 0.0
    max_d_jump = max(interface_d_metrics) if interface_d_metrics else 0.0

    status = "pass"
    if max_err > dirichlet_tol or max_v_jump > continuity_tol or max_d_jump > continuity_tol:
        margin = max(dirichlet_tol * 2.0, continuity_tol * 2.0)
        worst = max(max_err, max_v_jump, max_d_jump)
        status = "borderline" if worst <= margin else "fail"

    if "artifact_dir" in cfg and cfg["artifact_dir"]:
        path = cfg["artifact_dir"] / "gateB_bc.pt"  # type: ignore[operator]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "boundary_points": points.detach().cpu(),
                    "boundary_V": V_vals.detach().cpu(),
                    "max_err": max_err,
                    "interface_v_jump": interface_metrics,
                    "interface_d_jump": interface_d_metrics,
                },
                path,
            )
            evidence["residuals"] = str(path)
        except Exception:
            pass

    metrics = {
        "dirichlet_max_err": max_err,
        "interface_max_v_jump": max_v_jump,
        "interface_max_d_jump": max_d_jump,
        "sampled": float(points.shape[0]),
    }
    thresholds = {
        "dirichlet": dirichlet_tol,
        "continuity": continuity_tol,
    }
    if sampled:
        notes.append("sampled_boundary")
    if interface_metrics:
        notes.append("interface_checked")

    return GateResult(
        gate="B",
        status=status,
        metrics=metrics,
        thresholds=thresholds,
        evidence=evidence,
        oracle={"method": result.method, "fidelity": result.fidelity.value},
        notes=notes,
        config=cfg,
    )
