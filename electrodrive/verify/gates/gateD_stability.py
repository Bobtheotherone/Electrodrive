from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from . import GateResult, _assert_cuda_inputs
from ..oracle_types import OracleQuery, OracleResult


_GATE_D_NONFINITE_PENALTY = 1e30


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _ensure_eval(cfg: Dict[str, object]) -> Callable[[torch.Tensor], torch.Tensor]:
    fn = cfg.get("candidate_eval", None)
    if not callable(fn):
        raise ValueError("Gate D requires 'candidate_eval' callable")
    return fn  # type: ignore[return-value]


def run_gate(
    query: OracleQuery,
    result: OracleResult,
    *,
    config: Optional[Dict[str, object]] = None,
) -> GateResult:
    _assert_cuda_inputs(query, result)
    cfg = dict(config or {})
    candidate_eval = _ensure_eval(cfg)
    delta = float(cfg.get("delta", 1e-2))
    tolerance = float(cfg.get("stability_tol", 5e-2))
    seed = int(cfg.get("seed", 0))
    n_points = int(cfg.get("n_points", min(128, query.points.shape[0])))
    artifact_dir = cfg.get("artifact_dir", None)

    pts = query.points[:n_points].detach().clone()
    torch.manual_seed(seed)
    perturb = torch.randn_like(pts) * delta

    base_val = candidate_eval(pts)
    if isinstance(base_val, tuple):
        base_val = base_val[0]
    pert_val = candidate_eval(pts + perturb)
    if isinstance(pert_val, tuple):
        pert_val = pert_val[0]
    if not (torch.is_tensor(base_val) and torch.is_tensor(pert_val)):
        raise ValueError("Gate D candidate_eval must return tensor")
    if not (base_val.is_cuda and pert_val.is_cuda):
        raise ValueError("Gate D expects CUDA tensors")

    base64 = base_val.double()
    pert64 = pert_val.double()
    notes = []
    if not _finite_tensor(base64) or not _finite_tensor(pert64):
        rel_change = float(_GATE_D_NONFINITE_PENALTY)
        var_base = float(_GATE_D_NONFINITE_PENALTY)
        status = "fail"
        notes.append("nonfinite_inputs")
    else:
        diff = base64.flatten() - pert64.flatten()
        denom = torch.linalg.norm(base64.flatten()).clamp_min(1e-12)
        rel_change_t = torch.linalg.norm(diff) / denom
        var_base_t = torch.var(base64)
        if not _finite_tensor(rel_change_t) or not _finite_tensor(var_base_t):
            rel_change = float(_GATE_D_NONFINITE_PENALTY)
            var_base = float(_GATE_D_NONFINITE_PENALTY)
            status = "fail"
            notes.append("nonfinite_metrics")
        else:
            rel_change = float(rel_change_t.item())
            var_base = float(var_base_t.item())
            status = "pass" if rel_change <= tolerance else "borderline" if rel_change <= tolerance * 2.0 else "fail"

    evidence = {}
    if artifact_dir:
        path = artifact_dir / "gateD_stability.pt"  # type: ignore[operator]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"points": pts.cpu(), "perturb": perturb.cpu(), "base": base_val.detach().cpu(), "pert": pert_val.detach().cpu()}, path)
            evidence["perturbation"] = str(path)
        except Exception:
            pass

    metrics = {"relative_change": rel_change, "variance": var_base, "delta": delta}
    thresholds = {"stability_tol": tolerance}

    return GateResult(
        gate="D",
        status=status,
        metrics=metrics,
        thresholds=thresholds,
        evidence=evidence,
        oracle={"method": result.method, "fidelity": result.fidelity.value},
        notes=notes,
        config=cfg,
    )
