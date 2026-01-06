from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from electrodrive.gfdsl.compile import (
    CompileContext,
    GFDSLOperator,
    canonicalize,
    lower_program,
    validate_program,
)
from electrodrive.gfdsl.io import deserialize_program, serialize_program
from electrodrive.images.search import get_collocation_data, solve_sparse
from electrodrive.orchestration.parser import parse_spec
from electrodrive.utils.logging import JsonlLogger
from electrodrive.verify.utils import normalize_dtype, utc_now_iso
from electrodrive.verify.verifier import VerificationPlan, Verifier


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    program_path: Path
    summary_path: Path
    weights_path: Path
    weights_json_path: Path
    certificate_path: Path


class GFDSLCandidate:
    def __init__(self, eval_fn, summary: dict) -> None:
        self._eval_fn = eval_fn
        self.summary = dict(summary)

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        return self._eval_fn(points)

    def __str__(self) -> str:
        return json.dumps(self.summary, sort_keys=True, separators=(",", ":"))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GFDSL verification (GPU-first repo).")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_program(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return deserialize_program(payload)


def _load_spec(path: Path):
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return parse_spec(path)
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return parse_spec(payload)


def _normalize_gates(gates: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not gates:
        return ("A", "B")
    cleaned = []
    for gate in gates:
        gate = str(gate).strip().upper()
        if not gate:
            continue
        cleaned.append(gate)
    return tuple(cleaned) or ("A", "B")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _solution_summary(w: torch.Tensor) -> dict:
    if w.numel() == 0:
        return {"count": 0, "l2": 0.0, "max_abs": 0.0}
    return {
        "count": int(w.numel()),
        "l2": float(torch.linalg.norm(w).item()),
        "max_abs": float(torch.max(torch.abs(w)).item()),
    }


def run_gfdsl_verify(
    *,
    spec_path: Path,
    program_path: Path,
    out_dir: Path,
    dtype: torch.dtype = torch.float32,
    eval_backend: str = "operator",
    solver: str = "ista",
    reg_l1: float = 1e-3,
    seed: int = 0,
    gates: Optional[Sequence[str]] = None,
    n_points: Optional[int] = None,
    ratio_boundary: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    fp64_certify: bool = False,
    plan: Optional[VerificationPlan] = None,
) -> Tuple[RunArtifacts, object]:
    _require_cuda()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)

    _seed_all(seed)
    device = torch.device("cuda")
    if fp64_certify:
        dtype = torch.float64

    logger.info(
        "GFDSL verify run started.",
        spec=str(spec_path),
        program=str(program_path),
        out=str(out_dir),
        device=str(device),
        dtype=normalize_dtype(dtype),
        eval_backend=eval_backend,
        solver=solver,
        reg_l1=float(reg_l1),
        seed=int(seed),
        fp64_certify=bool(fp64_certify),
    )

    spec = _load_spec(spec_path)
    program = _load_program(program_path)

    ctx = CompileContext(spec=spec, device=device, dtype=dtype, eval_backend=eval_backend)
    validate_program(program, ctx)
    program = canonicalize(program)

    contrib = lower_program(program, ctx)

    colloc = get_collocation_data(
        spec,
        logger,
        device,
        dtype,
        return_is_boundary=True,
        n_points_override=n_points,
        ratio_override=ratio_boundary,
    )
    if len(colloc) == 2:
        X, V = colloc
        is_boundary = None
    else:
        X, V, is_boundary = colloc

    if X.numel() == 0:
        raise RuntimeError("Collocation batch is empty; cannot solve GFDSL coefficients.")

    fixed_term_fn = contrib.fixed_term
    if fixed_term_fn is not None:
        with torch.no_grad():
            fixed = fixed_term_fn(X)
        V_target = V - fixed
    else:
        fixed = None
        V_target = V

    operator = GFDSLOperator(contrib, points=X, device=device, dtype=dtype)
    k_slots = contrib.evaluator.K

    if k_slots > 0:
        solve_out = solve_sparse(
            operator,
            X,
            V_target,
            is_boundary,
            logger,
            reg_l1=float(reg_l1),
            solver=str(solver),
            max_iter=int(max_iter),
            tol=float(tol),
            return_stats=True,
        )
        weights, support, stats = solve_out
    else:
        weights = torch.zeros(0, device=device, dtype=dtype)
        support = []
        stats = {"solver": "none", "iters": 0, "converged": True}

    def _eval(points: torch.Tensor) -> torch.Tensor:
        pts = points.to(device=device)
        eval_dtype = pts.dtype
        if k_slots > 0:
            w_eval = weights.to(device=pts.device, dtype=eval_dtype)
            out = contrib.evaluator.matvec(w_eval, pts)
        else:
            out = torch.zeros(pts.shape[0], device=pts.device, dtype=eval_dtype)
        if fixed_term_fn is not None:
            out = out + fixed_term_fn(pts)
        return out

    program_payload = serialize_program(program, meta={"normalized_at": utc_now_iso()})
    program_json_path = out_dir / "gfdsl_program_normalized.json"
    _write_json(program_json_path, program_payload)

    summary_path = out_dir / "compile_summary.json"
    summary = {
        "slot_count": int(k_slots),
        "eval_backend": str(eval_backend),
        "device": str(device),
        "dtype": normalize_dtype(dtype),
        "has_fixed_term": fixed_term_fn is not None,
        "support": support,
        "solve_stats": stats,
        "solution": _solution_summary(weights),
    }
    _write_json(summary_path, summary)

    weights_path = out_dir / "solution_weights.pt"
    torch.save({"weights": weights.detach(), "fixed": fixed}, weights_path)
    weights_json_path = out_dir / "solution_weights.json"
    _write_json(weights_json_path, {"weights": weights.detach().cpu().tolist()})

    gate_order = _normalize_gates(gates)
    if plan is None:
        plan = VerificationPlan()
        plan.gate_order = list(gate_order)
        for gate in gate_order:
            plan.seeds[gate] = int(seed)
    else:
        plan.gate_order = list(plan.gate_order or gate_order)

    candidate_summary = {
        "kind": "gfdsl_program",
        "slot_count": int(k_slots),
        "weights": weights.detach().cpu().tolist(),
        "eval_backend": str(eval_backend),
    }
    candidate = GFDSLCandidate(_eval, candidate_summary)

    verifier = Verifier(out_root=out_dir)
    verify_dtype = torch.float64 if dtype != torch.float64 else dtype
    verify_points = torch.randn(
        max(plan.samples.get("A_interior", 128), 64),
        3,
        device=device,
        dtype=verify_dtype,
    )
    certificate = verifier.run(candidate, spec.to_json(), plan=plan, outdir=out_dir, points=verify_points)

    certificate_path = out_dir / "verification_certificate.json"
    _write_json(certificate_path, certificate.to_json())

    logger.info(
        "GFDSL verify run completed.",
        run_dir=str(out_dir),
        certificate=str(certificate_path),
        final_status=certificate.final_status,
    )

    artifacts = RunArtifacts(
        run_dir=out_dir,
        program_path=program_json_path,
        summary_path=summary_path,
        weights_path=weights_path,
        weights_json_path=weights_json_path,
        certificate_path=certificate_path,
    )
    return artifacts, certificate


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GFDSL JSON -> compile -> solve -> verify")
    parser.add_argument("--spec", required=True, help="Path to spec JSON/YAML")
    parser.add_argument("--program", required=True, help="Path to GFDSL program JSON")
    parser.add_argument("--out", required=True, help="Output run directory")
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "fp32", "fp64"], help="Evaluation dtype")
    parser.add_argument("--eval-backend", default="operator", choices=["operator", "dense"], help="Lowering backend")
    parser.add_argument("--solver", default="ista", help="Solver choice (ista/lista/implicit_lasso)")
    parser.add_argument("--reg-l1", type=float, default=1e-3, help="L1 regularization strength")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--gates", default="A,B", help="Comma-separated gate list (e.g. A,B)")
    parser.add_argument("--n-points", type=int, default=None, help="Override collocation point count")
    parser.add_argument("--ratio-boundary", type=float, default=None, help="Override boundary ratio")
    parser.add_argument("--max-iter", type=int, default=500, help="Solver max iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver tolerance")
    parser.add_argument("--fp64-certify", action="store_true", help="Force fp64 evaluation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    gates = [g for g in args.gates.split(",") if g.strip()]

    run_gfdsl_verify(
        spec_path=Path(args.spec),
        program_path=Path(args.program),
        out_dir=Path(args.out),
        dtype=dtype,
        eval_backend=str(args.eval_backend),
        solver=str(args.solver),
        reg_l1=float(args.reg_l1),
        seed=int(args.seed),
        gates=gates,
        n_points=args.n_points,
        ratio_boundary=args.ratio_boundary,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        fp64_certify=bool(args.fp64_certify),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
