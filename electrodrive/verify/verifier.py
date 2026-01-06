from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from electrodrive.learn.collocation import _infer_geom_type_from_spec
from electrodrive.orchestration.parser import CanonicalSpec
from .certificate import DiscoveryCertificate
from .gates import GateResult, gateA_pde, gateB_bc, gateC_asymptotics, gateD_stability, gateE_speed
from .oracle_backends import (
    F0AnalyticOracleBackend,
    F0CoarseBEMOracleBackend,
    F0CoarseSpectralOracleBackend,
    F1SommerfeldOracleBackend,
    F2BEMOracleBackend,
    F3SymbolicOracleBackend,
)
from .oracle_manager import OracleManager
from .oracle_registry import OracleBackend, OracleRegistry
from .oracle_types import CachePolicy, OracleFidelity, OracleQuantity, OracleQuery, OracleResult
from .utils import get_git_sha, normalize_dtype, sha256_json, utc_now_iso


@dataclass
class VerificationPlan:
    gate_order: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    seeds: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "oracle": 7})
    samples: Dict[str, int] = field(
        default_factory=lambda: {
            "A_interior": 128,
            "B_boundary": 96,
            "C_far": 96,
            "C_near": 96,
            "D_points": 128,
            "E_bench": 2048,
            "hard_points": 256,
        }
    )
    oracle_budget: Dict[str, object] = field(default_factory=lambda: {"allow_f1_auto": True, "allow_f2_auto": False, "fast_mode": True})
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "laplacian_linf": 5e-3,
            "bc_dirichlet": 1e-3,
            "bc_continuity": 5e-3,
            "slope_tol": 0.15,
            "stability": 5e-2,
            "min_speedup": 1.1,
        }
    )
    artifact_verbosity: str = "minimal"
    allow_f3: bool = True
    allow_f2: bool = False
    run_name: Optional[str] = None
    start_fidelity: OracleFidelity = OracleFidelity.F0

    def to_json(self) -> Dict[str, object]:
        return {
            "gate_order": list(self.gate_order),
            "seeds": dict(self.seeds),
            "samples": dict(self.samples),
            "oracle_budget": dict(self.oracle_budget),
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "artifact_verbosity": self.artifact_verbosity,
            "allow_f3": bool(self.allow_f3),
            "allow_f2": bool(self.allow_f2),
            "run_name": self.run_name,
            "start_fidelity": self.start_fidelity.value,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "VerificationPlan":
        plan = VerificationPlan()
        plan.gate_order = list(d.get("gate_order", plan.gate_order))
        plan.seeds = {str(k): int(v) for k, v in dict(d.get("seeds", plan.seeds)).items()}
        plan.samples = {str(k): int(v) for k, v in dict(d.get("samples", plan.samples)).items()}
        plan.oracle_budget = dict(d.get("oracle_budget", plan.oracle_budget))
        plan.thresholds = {str(k): float(v) for k, v in dict(d.get("thresholds", plan.thresholds)).items()}
        plan.artifact_verbosity = str(d.get("artifact_verbosity", plan.artifact_verbosity))
        plan.allow_f3 = bool(d.get("allow_f3", plan.allow_f3))
        plan.allow_f2 = bool(d.get("allow_f2", plan.allow_f2))
        plan.run_name = d.get("run_name", plan.run_name)  # type: ignore[assignment]
        if "start_fidelity" in d:
            try:
                val = d.get("start_fidelity", plan.start_fidelity.value)
                if isinstance(val, OracleFidelity):
                    plan.start_fidelity = val
                else:
                    plan.start_fidelity = OracleFidelity(str(val).upper())
            except Exception:
                plan.start_fidelity = OracleFidelity.F0
        return plan


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verification (GPU-first rule)")


def _default_registry() -> OracleRegistry:
    registry = OracleRegistry()
    shared_cache = None
    registry.register(F0AnalyticOracleBackend(shared_cache))
    registry.register(F0CoarseSpectralOracleBackend(cache=shared_cache))
    registry.register(F0CoarseBEMOracleBackend(cache=shared_cache))
    registry.register(F1SommerfeldOracleBackend())
    registry.register(F2BEMOracleBackend())
    registry.register(F3SymbolicOracleBackend())
    return registry


def _run_dir(base: Path, *, name: Optional[str] = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = name or stamp
    path = base / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def _backend_eval_fn(
    backend: OracleBackend,
    spec: Dict[str, object],
    quantity: OracleQuantity,
    cache_policy: CachePolicy,
    budget: Dict[str, object],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _eval(pts: torch.Tensor) -> torch.Tensor:
        query = OracleQuery(
            spec=spec,
            points=pts,
            quantity=quantity,
            requested_fidelity=backend.fidelity,
            device=str(pts.device),
            dtype=normalize_dtype(pts.dtype),
            cache_policy=cache_policy,
            budget=budget,
        )
        res = backend.evaluate(query)
        if res.V is not None:
            return res.V
        if res.E is not None:
            return res.E.norm(dim=1)
        return torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)

    return _eval


def _hard_points(spec: Dict[str, object], device: torch.device, dtype: torch.dtype, n: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    charges = spec.get("charges", []) or []
    if charges:
        centers = torch.tensor([c.get("pos", [0.0, 0.0, 0.0]) for c in charges], device=device, dtype=dtype)
        noise = torch.randn(n, 3, device=device, dtype=dtype) * 0.1
        idx = torch.randint(0, centers.shape[0], (n,), device=device)
        return centers[idx] + noise
    return torch.randn(n, 3, device=device, dtype=dtype)


def _artifact_dir(base: Path, gate: str) -> Path:
    out = base / "gates" / gate
    out.mkdir(parents=True, exist_ok=True)
    return out


class Verifier:
    def __init__(self, registry: Optional[OracleRegistry] = None, *, out_root: Path | str = "artifacts/verify_runs") -> None:
        self.registry = registry or _default_registry()
        self.out_root = Path(out_root)

    def _resolve_candidate_eval(self, candidate: Any, device: torch.device, dtype: torch.dtype) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        if isinstance(candidate, dict):
            if callable(candidate.get("eval_fn")):
                return candidate["eval_fn"]  # type: ignore[return-value]
            if "constant" in candidate:
                const_val = float(candidate["constant"])

                def _const(points: torch.Tensor) -> torch.Tensor:
                    return torch.full((points.shape[0],), const_val, device=points.device, dtype=points.dtype)

                return _const
        if callable(getattr(candidate, "evaluate", None)):
            return getattr(candidate, "evaluate")  # type: ignore[return-value]
        return None

    def _baseline_backend(self, spec: Dict[str, object], prefer_fidelity: OracleFidelity, query: OracleQuery) -> Optional[OracleBackend]:
        backend = self.registry.select_one(prefer_fidelity, query)
        if backend:
            return backend
        return None

    def _plan_name(self, plan: VerificationPlan, spec_digest: str, candidate_digest: str) -> str:
        if plan.run_name:
            return plan.run_name
        return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{spec_digest[:8]}_{candidate_digest[:8]}"

    def run(
        self,
        candidate: Dict[str, Any],
        spec: Dict[str, Any],
        plan: Optional[VerificationPlan] = None,
        *,
        points: Optional[torch.Tensor] = None,
        outdir: Optional[Path] = None,
    ) -> DiscoveryCertificate:
        _require_cuda()
        plan = plan or VerificationPlan()
        device = torch.device("cuda")
        dtype = torch.float32
        if points is None:
            n_pts = max(plan.samples.get("A_interior", 128), 64)
            points = torch.randn(n_pts, 3, device=device, dtype=dtype)
        else:
            if not points.is_cuda:
                raise ValueError("points must be CUDA tensor")
            points = points.to(device=device)
            dtype = points.dtype

        spec_digest = sha256_json(spec)
        try:
            candidate_digest = sha256_json(candidate) if isinstance(candidate, dict) else sha256_json(str(candidate))
        except Exception:
            candidate_digest = sha256_json(repr(candidate))
        run_dir = outdir or _run_dir(self.out_root, name=self._plan_name(plan, spec_digest, candidate_digest))

        manager = OracleManager(self.registry)

        base_query = OracleQuery(
            spec=spec,
            points=points,
            quantity=OracleQuantity.BOTH,
            requested_fidelity=plan.start_fidelity,
            device=str(device),
            dtype=normalize_dtype(dtype),
            seed=int(plan.seeds.get("oracle", 0)),
            budget=dict(plan.oracle_budget),
            cache_policy=CachePolicy.USE_CACHE,
        )

        oracle_runs: List[Dict[str, object]] = []
        gate_history: List[Dict[str, object]] = []

        def _record(result: OracleResult, backend: OracleBackend) -> None:
            rec = result.to_json()
            rec["backend"] = backend.name
            rec["fingerprint"] = backend.fingerprint()
            oracle_runs.append(rec)

        result, backend = manager.evaluate_with_backend(base_query)
        _record(result, backend)

        # Build candidate evaluator from current backend
        candidate_eval = self._resolve_candidate_eval(candidate, device, dtype) or _backend_eval_fn(
            backend, spec, base_query.quantity, base_query.cache_policy, base_query.budget
        )
        gate_results = self._run_gate_pass(
            spec=spec,
            query=base_query,
            result=result,
            candidate_eval=candidate_eval,
            plan=plan,
            run_dir=run_dir,
        )
        gate_history.append({"fidelity": backend.fidelity.value, "results": {g: r.to_json() for g, r in gate_results.items()}})

        # Escalation policy: rerun A-C with higher fidelity if borderline or layered media
        geom = _infer_geom_type_from_spec(CanonicalSpec.from_json(spec)) or "unknown"
        borderline = any(gate_results[g].status == "borderline" for g in ("A", "B", "C") if g in gate_results)
        dielectrics_present = bool(spec.get("dielectrics"))
        escalate_f1 = geom in ("layered", "plane_layer") or dielectrics_present or borderline
        if escalate_f1 and backend.fidelity != OracleFidelity.F1 and plan.oracle_budget.get("allow_f1_auto", True):
            query_f1 = OracleQuery(
                spec=spec,
                points=_hard_points(spec, device, dtype, plan.samples.get("hard_points", 128), plan.seeds.get("A", 0)),
                quantity=base_query.quantity,
                requested_fidelity=OracleFidelity.F1,
                device=str(device),
                dtype=normalize_dtype(dtype),
                seed=int(plan.seeds.get("oracle", 0)) + 11,
                budget=dict(plan.oracle_budget | {"allow_f1_auto": True}),
                cache_policy=CachePolicy.USE_CACHE,
            )
            try:
                res_f1, backend_f1 = manager.evaluate_with_backend(query_f1)
                _record(res_f1, backend_f1)
                cand_eval_f1 = candidate_eval
                gate_results_f1 = self._run_gate_pass(
                    spec=spec,
                    query=query_f1,
                    result=res_f1,
                    candidate_eval=cand_eval_f1,
                    plan=plan,
                    run_dir=run_dir,
                    gates=("A", "B", "C"),
                    suffix="f1",
                )
                gate_history.append(
                    {"fidelity": backend_f1.fidelity.value, "results": {g: r.to_json() for g, r in gate_results_f1.items()}}
                )
                gate_results.update({k: v for k, v in gate_results_f1.items()})
            except Exception:
                pass

        if plan.allow_f2 and backend.fidelity != OracleFidelity.F2 and all(
            gate_results[g].status != "fail" for g in ("A", "B", "C") if g in gate_results
        ):
            query_f2 = OracleQuery(
                spec=spec,
                points=_hard_points(spec, device, dtype, plan.samples.get("hard_points", 128), plan.seeds.get("B", 1)),
                quantity=base_query.quantity,
                requested_fidelity=OracleFidelity.F2,
                device=str(device),
                dtype=normalize_dtype(torch.float64),
                seed=int(plan.seeds.get("oracle", 0)) + 17,
                budget=dict(plan.oracle_budget | {"allow_f2_auto": True}),
                cache_policy=CachePolicy.USE_CACHE,
            )
            try:
                res_f2, backend_f2 = manager.evaluate_with_backend(query_f2)
                _record(res_f2, backend_f2)
                cand_eval_f2 = candidate_eval
                gate_results_f2 = self._run_gate_pass(
                    spec=spec,
                    query=query_f2,
                    result=res_f2,
                    candidate_eval=cand_eval_f2,
                    plan=plan,
                    run_dir=run_dir,
                    gates=("A", "B", "C"),
                    suffix="f2",
                )
                gate_history.append(
                    {"fidelity": backend_f2.fidelity.value, "results": {g: r.to_json() for g, r in gate_results_f2.items()}}
                )
                gate_results.update({k: v for k, v in gate_results_f2.items()})
            except Exception:
                pass

        # Stability and speed gates on final candidate evaluator (opt-in via gate_order).
        run_gate_d = "D" in plan.gate_order
        run_gate_e = "E" in plan.gate_order

        if run_gate_d:
            gate_results["D"] = gateD_stability.run_gate(
                base_query,
                result,
                config={
                    "candidate_eval": candidate_eval,
                    "delta": float(plan.thresholds.get("stability", 5e-2)),
                    "stability_tol": float(plan.thresholds.get("stability", 5e-2)),
                    "n_points": plan.samples.get("D_points", 128),
                    "seed": plan.seeds.get("D", 3),
                    "artifact_dir": _artifact_dir(run_dir, "D"),
                },
            )

        if run_gate_e:
            prefers_layered = geom.startswith("layer") or geom == "plane_layer"
            baseline_backend = self._baseline_backend(
                spec,
                OracleFidelity.F1 if prefers_layered else OracleFidelity.F2,
                base_query,
            )
            if baseline_backend is None:
                baseline_eval = candidate_eval
            else:
                baseline_eval = _backend_eval_fn(
                    baseline_backend,
                    spec,
                    base_query.quantity,
                    base_query.cache_policy,
                    base_query.budget,
                )
            gate_results["E"] = gateE_speed.run_gate(
                base_query,
                result,
                config={
                    "candidate_eval": candidate_eval,
                    "baseline_eval": baseline_eval,
                    "n_bench": plan.samples.get("E_bench", 2048),
                    "min_speedup": plan.thresholds.get("min_speedup", 1.1),
                    "prereq_pass": all(
                        gate_results[g].status == "pass" for g in ("A", "B", "C") if g in gate_results
                    ),
                    "artifact_dir": _artifact_dir(run_dir, "E"),
                },
            )

        final_status = self._final_status(gate_results)
        reasons = self._reasons(gate_results)

        hardware = {
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            "device": str(device),
        }

        gates_json = {k: v.to_json() for k, v in gate_results.items()}
        gates_json["history"] = gate_history

        certificate = DiscoveryCertificate(
            spec_digest=spec_digest,
            candidate_digest=candidate_digest,
            git_sha=get_git_sha(),
            hardware=hardware,
            oracle_runs=oracle_runs,
            gates=gates_json,
            final_status=final_status,
            reasons=reasons,
            attachments=[],
        )

        dashboard = {
            "status": final_status,
            "timestamp": utc_now_iso(),
            "run_dir": str(run_dir),
            "gates": gates_json,
            "oracle_runs": oracle_runs,
            "plan": plan.to_json(),
        }
        (run_dir / "discovery_certificate.json").write_text(json.dumps(certificate.to_json(), indent=2), encoding="utf-8")
        (run_dir / "verify_dashboard.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "verify_dashboard.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")

        return certificate

    def _run_gate_pass(
        self,
        *,
        spec: Dict[str, Any],
        query: OracleQuery,
        result: OracleResult,
        candidate_eval: Callable[[torch.Tensor], torch.Tensor],
        plan: VerificationPlan,
        run_dir: Path,
        gates: Tuple[str, ...] | None = None,
        suffix: str = "",
    ) -> Dict[str, GateResult]:
        gates = gates or tuple(plan.gate_order)
        gate_results: Dict[str, GateResult] = {}
        for gate in gates:
            art_dir = _artifact_dir(run_dir, gate)
            if suffix:
                art_dir = art_dir / suffix
            if gate == "A":
                gate_results["A"] = gateA_pde.run_gate(
                    query,
                    result,
                    config={
                        "seed": plan.seeds.get("A", 0),
                        "n_interior": plan.samples.get("A_interior", 128),
                        "exclusion_radius": 5e-2,
                        "linf_tol": plan.thresholds.get("laplacian_linf", 5e-3),
                        "l2_tol": plan.thresholds.get("laplacian_linf", 5e-3),
                        "p95_tol": plan.thresholds.get("laplacian_linf", 5e-3),
                        "prefer_autograd": True,
                        "autograd_max_samples": plan.samples.get("A_interior", 128),
                        "spec": spec,
                        "candidate_eval": candidate_eval,
                        "artifact_dir": art_dir,
                    },
                )
            elif gate == "B":
                gate_results["B"] = gateB_bc.run_gate(
                    query,
                    result,
                    config={
                        "seed": plan.seeds.get("B", 1),
                        "n_samples": plan.samples.get("B_boundary", 96),
                        "tolerance": plan.thresholds.get("bc_dirichlet", 1e-3),
                        "continuity_tol": plan.thresholds.get("bc_continuity", 5e-3),
                        "eval_fn": candidate_eval,
                        "artifact_dir": art_dir,
                    },
                )
            elif gate == "C":
                gate_results["C"] = gateC_asymptotics.run_gate(
                    query,
                    result,
                    config={
                        "seed": plan.seeds.get("C", 2),
                        "n_far": plan.samples.get("C_far", 96),
                        "n_near": plan.samples.get("C_near", 96),
                        "far_radius": 10.0,
                        "near_radius": 0.5,
                        "slope_tol": plan.thresholds.get("slope_tol", 0.15),
                        "candidate_eval": candidate_eval,
                        "artifact_dir": art_dir,
                    },
                )
            elif gate == "D":
                gate_results["D"] = gateD_stability.run_gate(
                    query,
                    result,
                    config={
                        "candidate_eval": candidate_eval,
                        "delta": float(plan.thresholds.get("stability", 5e-2)),
                        "stability_tol": float(plan.thresholds.get("stability", 5e-2)),
                        "n_points": plan.samples.get("D_points", 128),
                        "seed": plan.seeds.get("D", 3),
                        "artifact_dir": art_dir,
                    },
                )
            elif gate == "E":
                # gate E handled outside in run()
                continue
        return gate_results

    def _final_status(self, gates: Dict[str, GateResult]) -> str:
        if any(g.status == "fail" for g in gates.values()):
            return "fail"
        if any(g.status == "borderline" for g in gates.values()):
            return "borderline"
        return "pass"

    def _reasons(self, gates: Dict[str, GateResult]) -> List[str]:
        reasons: List[str] = []
        for name, gate in gates.items():
            if gate.status in ("fail", "borderline"):
                reasons.append(f"gate_{name}_{gate.status}")
        return reasons
