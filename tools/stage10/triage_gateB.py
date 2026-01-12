from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

from electrodrive.experiments.run_discovery import SpecSampler
from electrodrive.images.basis import ImageBasisElement
from electrodrive.images.search import ImageSystem
from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.verify.gates import gateB_bc
from electrodrive.verify.verifier import VerificationPlan


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _plan_from_verifier(verify_path: Path) -> VerificationPlan:
    dash_path = verify_path / "verify_dashboard.json"
    if dash_path.exists():
        payload = _load_json(dash_path)
        plan = payload.get("plan")
        if isinstance(plan, dict):
            return VerificationPlan.from_json(plan)
    return VerificationPlan()


def _specs_for_run(cfg: Dict[str, Any], generations: int) -> Dict[int, CanonicalSpec]:
    # CPU-only: spec sampling is lightweight and deterministic.
    seed = int(cfg.get("seed", 0))
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    fixed_spec = bool(run_cfg.get("fixed_spec", False))
    fixed_spec_index = int(run_cfg.get("fixed_spec_index", 0))
    if fixed_spec_index < 0:
        fixed_spec_index = 0
    spec_cfg = cfg.get("spec", {}) if isinstance(cfg.get("spec", {}), dict) else {}
    sampler = SpecSampler(spec_cfg, seed)

    if fixed_spec:
        fixed_spec_obj = None
        for _ in range(fixed_spec_index + 1):
            fixed_spec_obj = sampler.sample()
        if fixed_spec_obj is None:
            raise RuntimeError("fixed_spec enabled but no spec sampled.")
        return {gen: fixed_spec_obj for gen in range(generations)}

    return {gen: sampler.sample() for gen in range(generations)}


def _load_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    cert_dir = run_dir / "artifacts" / "certificates"
    if not cert_dir.exists():
        return []
    summaries = []
    for path in sorted(cert_dir.glob("*_summary.json")):
        payload = _load_json(path)
        payload["_summary_path"] = str(path)
        summaries.append(payload)
    return summaries


def _select_candidates(
    summaries: List[Dict[str, Any]],
    *,
    best_partial_path: str | None,
    top_n: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    by_path: Dict[str, Dict[str, Any]] = {s["_summary_path"]: s for s in summaries}

    if best_partial_path and best_partial_path in by_path:
        selected.append(by_path[best_partial_path])

    ranked = sorted(
        summaries,
        key=lambda s: (int(s.get("rank", 0)), int(s.get("generation", 0))),
    )
    for cand in ranked:
        if cand in selected:
            continue
        selected.append(cand)
        if len(selected) >= top_n + (1 if best_partial_path else 0):
            break
    return selected


def _reference_eval(spec: CanonicalSpec, pts: torch.Tensor) -> torch.Tensor:
    dielectrics = getattr(spec, "dielectrics", None) or []
    if not dielectrics:
        return torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)
    return compute_layered_reference_potential(spec, pts, device=pts.device, dtype=pts.dtype)


def _gateB_breakdown(
    *,
    spec: CanonicalSpec,
    eval_fn: callable,
    plan: VerificationPlan,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    dirichlet_tol = float(plan.thresholds.get("bc_dirichlet", 1e-3))
    continuity_tol = float(plan.thresholds.get("bc_continuity", 5e-3))
    seed = int(plan.seeds.get("B", 1))
    n_samples = int(plan.samples.get("B_boundary", 96))
    interface_delta = 5e-3

    def _ensure_cuda(tensor: torch.Tensor, label: str) -> torch.Tensor:
        if not tensor.is_cuda:
            raise ValueError(f"{label} must be CUDA tensor")
        return tensor

    def _grad(points: torch.Tensor) -> torch.Tensor:
        pts = points.detach().clone().requires_grad_(True)
        vals = _ensure_cuda(eval_fn(pts), "gateB_eval")
        grad = torch.autograd.grad(vals, pts, grad_outputs=torch.ones_like(vals), create_graph=False)[0]
        return grad.detach()

    has_conductors = bool(getattr(spec, "conductors", None) or [])
    target_potential = 0.0
    if has_conductors:
        target_potential = float(getattr(spec, "conductors")[0].get("potential", 0.0))

    dirichlet_max_err = 0.0
    boundary_samples = 0
    notes: List[str] = []
    if has_conductors:
        boundary_pts = gateB_bc._sample_boundary(spec, device, dtype, n_samples, seed=seed)
        if boundary_pts is None:
            notes.append("no_boundary_samples")
        else:
            boundary_samples = int(boundary_pts.shape[0])
            vals = _ensure_cuda(eval_fn(boundary_pts), "gateB_boundary_eval")
            if vals.numel() > 0:
                dirichlet_max_err = float(torch.max(torch.abs(vals - target_potential)).item())
    else:
        notes.append("no_conductors")

    interface_v_jumps: List[float] = []
    interface_d_jumps: List[float] = []
    dielectrics = getattr(spec, "dielectrics", None) or []
    if dielectrics:
        interfaces = gateB_bc._interfaces(dielectrics)
        if interfaces:
            # SobolEngine draws on CPU; move to CUDA immediately to keep eval GPU-resident.
            xy = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed + 1).draw(n_samples).to(
                device=device, dtype=dtype
            )
            xy = (xy - 0.5) * 2.0
            normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            for z_val, eps_up, eps_down in interfaces:
                pts_upper = torch.stack(
                    [xy[:, 0], xy[:, 1], torch.full((n_samples,), z_val + interface_delta, device=device, dtype=dtype)],
                    dim=1,
                )
                pts_lower = torch.stack(
                    [xy[:, 0], xy[:, 1], torch.full((n_samples,), z_val - interface_delta, device=device, dtype=dtype)],
                    dim=1,
                )
                v_up = _ensure_cuda(eval_fn(pts_upper), "gateB_interface_eval")
                v_low = _ensure_cuda(eval_fn(pts_lower), "gateB_interface_eval")
                grad_up = _grad(pts_upper)
                grad_low = _grad(pts_lower)
                d_jump = torch.abs(eps_up * torch.sum(grad_up * normal, dim=1) - eps_down * torch.sum(grad_low * normal, dim=1))
                v_jump = torch.abs(v_up - v_low)
                interface_v_jumps.append(float(torch.max(v_jump).item()))
                interface_d_jumps.append(float(torch.max(d_jump).item()))
        else:
            notes.append("no_interfaces")
    else:
        notes.append("no_dielectrics")

    max_v_jump = max(interface_v_jumps) if interface_v_jumps else 0.0
    max_d_jump = max(interface_d_jumps) if interface_d_jumps else 0.0
    max_residual = max(dirichlet_max_err, max_v_jump, max_d_jump)
    dominant_term = "dirichlet"
    if max_v_jump >= dirichlet_max_err and max_v_jump >= max_d_jump:
        dominant_term = "interface_potential"
    elif max_d_jump >= dirichlet_max_err and max_d_jump >= max_v_jump:
        dominant_term = "interface_displacement"

    status = "pass"
    if max_residual > max(dirichlet_tol, continuity_tol):
        margin = max(dirichlet_tol * 2.0, continuity_tol * 2.0)
        status = "borderline" if max_residual <= margin else "fail"

    return {
        "status": status,
        "dirichlet_max_err": dirichlet_max_err,
        "interface_max_v_jump": max_v_jump,
        "interface_max_d_jump": max_d_jump,
        "max_residual": max_residual,
        "dominant_term": dominant_term,
        "boundary_samples": boundary_samples,
        "interface_samples": n_samples if interface_v_jumps else 0,
        "interface_delta": interface_delta,
        "dirichlet_tol": dirichlet_tol,
        "continuity_tol": continuity_tol,
        "notes": notes,
    }


def _format_md(results: Dict[str, Any]) -> str:
    lines = ["# Gate B triage", ""]
    lines.append(f"run_dir: {results.get('run_dir')}")
    lines.append(f"run_id: {results.get('run_id')}")
    lines.append("")
    for cand in results.get("candidates", []):
        lines.append(f"## gen {cand.get('generation')} rank {cand.get('rank')}")
        lines.append(f"summary: {cand.get('summary_path')}")
        lines.append(f"verification: {cand.get('verification_path', 'none')}")
        lines.append("")
        lines.append("| mode | dirichlet_max_err | interface_max_v_jump | interface_max_d_jump | max_residual | dominant | status |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for mode, metrics in cand.get("modes", {}).items():
            lines.append(
                "| {mode} | {dirichlet_max_err:.3e} | {interface_max_v_jump:.3e} | {interface_max_d_jump:.3e} | {max_residual:.3e} | {dominant_term} | {status} |".format(
                    mode=mode,
                    dirichlet_max_err=float(metrics.get("dirichlet_max_err", 0.0)),
                    interface_max_v_jump=float(metrics.get("interface_max_v_jump", 0.0)),
                    interface_max_d_jump=float(metrics.get("interface_max_d_jump", 0.0)),
                    max_residual=float(metrics.get("max_residual", 0.0)),
                    dominant_term=str(metrics.get("dominant_term", "none")),
                    status=str(metrics.get("status", "unknown")),
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage10 Gate B triage (reference +/-).")
    parser.add_argument("run_dir", type=Path, help="Run directory with artifacts/certificates.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top-ranked candidates to triage.")
    parser.add_argument("--out-root", type=Path, default=Path("stage10/audit"), help="Audit root output directory.")
    args = parser.parse_args()

    run_dir = args.run_dir
    run_id = run_dir.name

    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for Gate B triage.")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config.yaml in {run_dir}")
    cfg = _load_yaml(cfg_path)
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    generations = int(run_cfg.get("generations", 1))
    specs = _specs_for_run(cfg, generations)

    summaries = _load_summaries(run_dir)
    if not summaries:
        raise SystemExit("No certificate summaries found.")

    analysis_summary = run_dir / "analysis" / "analysis_summary.json"
    best_partial_path = None
    if analysis_summary.exists():
        payload = _load_json(analysis_summary)
        best_partial = payload.get("best_partial", {})
        if isinstance(best_partial, dict):
            best_partial_path = best_partial.get("summary_path")

    selected = _select_candidates(summaries, best_partial_path=best_partial_path, top_n=max(1, args.top_n))

    device = torch.device("cuda")
    dtype = torch.float32
    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "candidates": [],
    }

    for cand in selected:
        gen = int(cand.get("generation", 0))
        spec = specs.get(gen)
        if spec is None:
            continue
        elements_data = cand.get("elements", []) or []
        weights_data = cand.get("weights", []) or []
        if not elements_data or not weights_data:
            continue

        elements = [
            ImageBasisElement.deserialize(elem, device=device, dtype=dtype)
            for elem in elements_data
        ]
        weights = torch.tensor(weights_data, device=device, dtype=dtype)
        if weights.numel() != len(elements):
            continue
        system = ImageSystem(elements, weights)

        def _candidate_eval(pts: torch.Tensor) -> torch.Tensor:
            out = system.potential(pts)
            if not out.is_cuda:
                raise ValueError("candidate_eval returned CPU tensor")
            return out

        def _plus_ref(pts: torch.Tensor) -> torch.Tensor:
            return _candidate_eval(pts) + _reference_eval(spec, pts)

        def _minus_ref(pts: torch.Tensor) -> torch.Tensor:
            return _candidate_eval(pts) - _reference_eval(spec, pts)

        verification_path = cand.get("verification", {}).get("path", None) if isinstance(cand.get("verification", {}), dict) else None
        plan = _plan_from_verifier(Path(verification_path)) if verification_path else VerificationPlan()

        modes = {
            "candidate_only": _gateB_breakdown(spec=spec, eval_fn=_candidate_eval, plan=plan, device=device, dtype=dtype),
            "candidate_plus_reference": _gateB_breakdown(spec=spec, eval_fn=_plus_ref, plan=plan, device=device, dtype=dtype),
            "candidate_minus_reference": _gateB_breakdown(spec=spec, eval_fn=_minus_ref, plan=plan, device=device, dtype=dtype),
        }

        results["candidates"].append(
            {
                "generation": gen,
                "rank": int(cand.get("rank", 0)),
                "summary_path": cand.get("_summary_path"),
                "verification_path": verification_path,
                "spec_digest": cand.get("spec_digest"),
                "modes": modes,
            }
        )

    out_dir = args.out_root / run_id / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gateB_triage.json"
    md_path = out_dir / "gateB_triage.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_format_md(results), encoding="utf-8")

    print(str(json_path))
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
