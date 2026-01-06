from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import torch

from .oracle_backends import (
    F0AnalyticOracleBackend,
    F0CoarseBEMOracleBackend,
    F0CoarseSpectralOracleBackend,
    F1SommerfeldOracleBackend,
    F2BEMOracleBackend,
)
from .oracle_manager import OracleManager
from .oracle_registry import OracleRegistry
from .oracle_types import CachePolicy, OracleFidelity, OracleQuantity, OracleQuery, OracleResult
from .utils import get_git_sha, normalize_dtype, sha256_json


EvalFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LadderConfig:
    n_points_f0: int = 128
    n_points_f1: int = 256
    n_points_f2: int = 512
    f0_threshold: float = 1e-2
    f1_threshold: float = 5e-3
    allow_f1: bool = True
    allow_f2: bool = False
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    seed: int = 7
    oracle_budget: Dict[str, object] = field(
        default_factory=lambda: {"allow_f1_auto": True, "allow_f2_auto": False, "fast_mode": True}
    )
    quantity: OracleQuantity = OracleQuantity.V


@dataclass
class LadderRun:
    fidelity: str
    backend: str
    n_points: int
    rel_l2: float
    max_abs: float
    latency_ms: Dict[str, float]
    passed: bool

    def to_json(self) -> Dict[str, object]:
        return {
            "fidelity": self.fidelity,
            "backend": self.backend,
            "n_points": int(self.n_points),
            "rel_l2": float(self.rel_l2),
            "max_abs": float(self.max_abs),
            "latency_ms": {k: float(v) for k, v in self.latency_ms.items()},
            "passed": bool(self.passed),
        }


@dataclass
class LadderResult:
    candidate_hash: str
    program_hash: str
    program: object
    parameters: object
    weights: object
    git_sha: str
    runs: List[LadderRun]
    structure_gate: Dict[str, object]

    def to_json(self) -> Dict[str, object]:
        return {
            "candidate_hash": self.candidate_hash,
            "program_hash": self.program_hash,
            "program": self.program,
            "parameters": self.parameters,
            "weights": self.weights,
            "git_sha": self.git_sha,
            "runs": [r.to_json() for r in self.runs],
            "structure_gate": dict(self.structure_gate),
        }


def build_default_registry() -> OracleRegistry:
    registry = OracleRegistry()
    shared_cache = None
    registry.register(F0AnalyticOracleBackend(shared_cache))
    registry.register(F0CoarseSpectralOracleBackend(cache=shared_cache))
    registry.register(F0CoarseBEMOracleBackend(cache=shared_cache))
    registry.register(F1SommerfeldOracleBackend())
    registry.register(F2BEMOracleBackend())
    return registry


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_points(n: int, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(n, 3, device=device, dtype=dtype, generator=gen)


def _resolve_eval_fn(candidate: object, eval_fn: Optional[EvalFn]) -> Optional[EvalFn]:
    if eval_fn is not None:
        return eval_fn
    if isinstance(candidate, dict):
        fn = candidate.get("eval_fn")
        if callable(fn):
            return fn  # type: ignore[return-value]
    if callable(getattr(candidate, "evaluate", None)):
        return getattr(candidate, "evaluate")  # type: ignore[return-value]
    if callable(getattr(candidate, "potential", None)):
        return getattr(candidate, "potential")  # type: ignore[return-value]
    return None


def _extract_reference(result: OracleResult) -> torch.Tensor:
    if result.V is not None:
        return result.V
    if result.E is not None:
        return result.E.norm(dim=1)
    raise ValueError("Oracle result did not return V or E.")


def _structure_gate(candidate: object) -> Dict[str, object]:
    missing: List[str] = []
    if isinstance(candidate, dict):
        for key in ("program", "elements", "weights"):
            if key not in candidate:
                missing.append(key)
    status = "pass" if not missing else "fail"
    return {"status": status, "missing": missing}


def run_ladder(
    candidate: Dict[str, object],
    spec: Dict[str, object],
    *,
    eval_fn: Optional[EvalFn] = None,
    config: Optional[LadderConfig] = None,
    registry: Optional[OracleRegistry] = None,
) -> LadderResult:
    cfg = config or LadderConfig()
    device = _resolve_device(cfg.device)
    dtype = cfg.dtype
    manager = OracleManager(registry or build_default_registry())

    candidate_hash = sha256_json(candidate)
    program = candidate.get("program")
    if program is None:
        program = candidate.get("program_template") or candidate.get("template") or []
    program_hash = sha256_json(program)
    parameters = candidate.get("elements") or candidate.get("params") or {}
    weights = candidate.get("weights")

    gate = _structure_gate(candidate)
    if gate["status"] != "pass":
        return LadderResult(
            candidate_hash=candidate_hash,
            program_hash=program_hash,
            program=program,
            parameters=parameters,
            weights=weights,
            git_sha=get_git_sha(),
            runs=[],
            structure_gate=gate,
        )

    candidate_eval = _resolve_eval_fn(candidate, eval_fn)
    if candidate_eval is None:
        raise ValueError("Candidate evaluator is required for ladder evaluation.")

    runs: List[LadderRun] = []

    def _run_fidelity(
        fidelity: OracleFidelity,
        n_points: int,
        threshold: float,
        seed: int,
    ) -> Tuple[bool, LadderRun]:
        pts = _sample_points(n_points, device=device, dtype=dtype, seed=seed)
        query = OracleQuery(
            spec=spec,
            points=pts,
            quantity=cfg.quantity,
            requested_fidelity=fidelity,
            device=str(device),
            dtype=normalize_dtype(dtype),
            seed=int(seed),
            budget=dict(cfg.oracle_budget),
            cache_policy=CachePolicy.USE_CACHE,
        )
        t0 = time.perf_counter()
        result, backend = manager.evaluate_with_backend(query)
        t1 = time.perf_counter()
        pred = candidate_eval(pts)
        t2 = time.perf_counter()
        ref = _extract_reference(result)
        diff = pred - ref
        denom = torch.linalg.norm(ref).clamp_min(1e-12)
        rel_l2 = float(torch.linalg.norm(diff).item() / denom.item())
        max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
        passed = rel_l2 <= threshold
        run = LadderRun(
            fidelity=fidelity.value,
            backend=backend.name,
            n_points=int(n_points),
            rel_l2=rel_l2,
            max_abs=max_abs,
            latency_ms={
                "oracle": (t1 - t0) * 1000.0,
                "candidate": (t2 - t1) * 1000.0,
                "total": (t2 - t0) * 1000.0,
            },
            passed=passed,
        )
        return passed, run

    ok_f0, run_f0 = _run_fidelity(OracleFidelity.F0, cfg.n_points_f0, cfg.f0_threshold, cfg.seed)
    runs.append(run_f0)

    if cfg.allow_f1 and ok_f0:
        ok_f1, run_f1 = _run_fidelity(OracleFidelity.F1, cfg.n_points_f1, cfg.f1_threshold, cfg.seed + 11)
        runs.append(run_f1)
        if cfg.allow_f2 and ok_f1:
            _, run_f2 = _run_fidelity(OracleFidelity.F2, cfg.n_points_f2, cfg.f1_threshold, cfg.seed + 17)
            runs.append(run_f2)

    return LadderResult(
        candidate_hash=candidate_hash,
        program_hash=program_hash,
        program=program,
        parameters=parameters,
        weights=weights,
        git_sha=get_git_sha(),
        runs=runs,
        structure_gate=gate,
    )
