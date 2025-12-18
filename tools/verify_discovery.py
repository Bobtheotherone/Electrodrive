from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from electrodrive.verify.verifier import VerificationPlan, Verifier
from electrodrive.verify.utils import sha256_json
from electrodrive.verify.oracle_types import OracleFidelity
from electrodrive.verify.oracle_registry import OracleRegistry
from electrodrive.verify.oracle_backends import (
    F0AnalyticOracleBackend,
    F0CoarseSpectralOracleBackend,
    F0CoarseBEMOracleBackend,
    F1SommerfeldOracleBackend,
    F2BEMOracleBackend,
    F3SymbolicOracleBackend,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verification (GPU-first rule)")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_points(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Points path does not exist: {path}")
    arr_data: Any
    if path.suffix.lower() == ".npy":
        arr_data = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "points" in data:
                arr_data = data["points"]
            elif "arr_0" in data:
                arr_data = data["arr_0"]
            elif len(data.files) == 1:
                arr_data = data[data.files[0]]
            else:
                raise ValueError("NPZ points file must contain array 'points' or a single unnamed array")
    else:
        arr_data = json.loads(path.read_text(encoding="utf-8"))
    arr = torch.as_tensor(arr_data)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Points array must have shape [N, 3], got {tuple(arr.shape)}")
    if not torch.is_floating_point(arr):
        arr = arr.float()
    return arr.to(device=device, dtype=dtype)


def _sample_default_points(spec: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = 256
    conductors = spec.get("conductors", [])
    pts = torch.randn(n, 3, device=device, dtype=dtype)
    if conductors:
        c = conductors[0]
        ctype = c.get("type")
        if ctype == "plane":
            z = float(c.get("z", 0.0))
            pts[: n // 4, 2] = z
        elif ctype == "sphere":
            r = float(c.get("radius", 1.0))
            center = torch.tensor(c.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
            theta = torch.rand(n // 4, device=device, dtype=dtype) * 2.0 * torch.pi
            phi = torch.rand(n // 4, device=device, dtype=dtype) * torch.pi
            x = center[0] + r * torch.sin(phi) * torch.cos(theta)
            y = center[1] + r * torch.sin(phi) * torch.sin(theta)
            z = center[2] + r * torch.cos(phi)
            pts[: n // 4] = torch.stack([x, y, z], dim=1)
    return pts


def _load_plan(args: argparse.Namespace) -> VerificationPlan:
    if args.plan:
        plan_blob = Path(args.plan)
        if plan_blob.exists():
            data = _load_json(plan_blob)
        else:
            data = json.loads(args.plan)
        plan = VerificationPlan.from_json(data)
    else:
        plan = VerificationPlan()
        plan.samples.update({"A_interior": 48, "B_boundary": 48, "C_far": 48, "C_near": 48, "E_bench": 256})
        plan.thresholds["min_speedup"] = 0.5
    if args.samples_interior is not None:
        plan.samples["A_interior"] = int(args.samples_interior)
    if args.samples_boundary is not None:
        plan.samples["B_boundary"] = int(args.samples_boundary)
    if args.samples_far is not None:
        plan.samples["C_far"] = int(args.samples_far)
    if args.samples_near is not None:
        plan.samples["C_near"] = int(args.samples_near)
    if args.bench_samples is not None:
        plan.samples["E_bench"] = int(args.bench_samples)
    if args.min_speedup is not None:
        plan.thresholds["min_speedup"] = float(args.min_speedup)
    if args.allow_f2:
        plan.allow_f2 = True
        plan.oracle_budget["allow_f2_auto"] = True
    if args.allow_f1:
        plan.oracle_budget["allow_f1_auto"] = True
    plan.start_fidelity = {
        "auto": OracleFidelity.AUTO,
        "F0": OracleFidelity.F0,
        "F1": OracleFidelity.F1,
        "F2": OracleFidelity.F2,
        "F3": OracleFidelity.F3,
    }[args.oracle]
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Step-8 Truth Engine driver (multi-fidelity + gates).")
    ap.add_argument("--spec", required=True, help="Path to spec JSON")
    ap.add_argument("--candidate", required=False, help="Path to candidate JSON (optional)")
    ap.add_argument("--points", required=False, help="Optional path to evaluation points (JSON, NPY, or NPZ)")
    ap.add_argument("--oracle", required=False, default="auto", choices=["auto", "F0", "F1", "F2", "F3"], help="Oracle fidelity to start with")
    ap.add_argument("--outdir", required=False, default=None, help="Output directory root")
    ap.add_argument("--plan", required=False, help="Plan JSON (path or inline JSON)")
    ap.add_argument("--samples-interior", dest="samples_interior", type=int, default=None, help="Override interior sample count")
    ap.add_argument("--samples-boundary", dest="samples_boundary", type=int, default=None, help="Override boundary sample count")
    ap.add_argument("--samples-far", dest="samples_far", type=int, default=None, help="Override far-field sample count")
    ap.add_argument("--samples-near", dest="samples_near", type=int, default=None, help="Override near-singularity sample count")
    ap.add_argument("--bench-samples", dest="bench_samples", type=int, default=None, help="Override speed bench sample count")
    ap.add_argument("--min-speedup", dest="min_speedup", type=float, default=None, help="Minimum speedup for Gate E")
    ap.add_argument("--allow-f2", dest="allow_f2", action="store_true", help="Allow escalation to F2")
    ap.add_argument("--allow-f1", dest="allow_f1", action="store_true", help="Allow escalation to F1")
    ap.add_argument("--full-registry", dest="full_registry", action="store_true", help="Include slower BEM backends")
    args = ap.parse_args()

    _require_cuda()

    spec = _load_json(Path(args.spec))
    candidate: Dict[str, Any] = {}
    if args.candidate:
        candidate_path = Path(args.candidate)
        if candidate_path.exists():
            candidate = _load_json(candidate_path)
        else:
            candidate = json.loads(args.candidate)

    device = torch.device("cuda")
    dtype = torch.float32
    points: torch.Tensor | None = None
    if args.points:
        points = _load_points(Path(args.points), device=device, dtype=dtype)
    else:
        points = _sample_default_points(spec, device=device, dtype=dtype)

    plan = _load_plan(args)
    if args.outdir:
        out_root = Path(args.outdir)
    else:
        out_root = Path("artifacts/verify_runs")

    registry = OracleRegistry()
    registry.register(F0AnalyticOracleBackend())
    registry.register(F0CoarseSpectralOracleBackend())
    registry.register(F1SommerfeldOracleBackend())
    registry.register(F3SymbolicOracleBackend())
    if args.full_registry:
        registry.register(F0CoarseBEMOracleBackend())
        registry.register(F2BEMOracleBackend())

    verifier = Verifier(registry=registry, out_root=out_root)
    certificate = verifier.run(candidate, spec, plan, points=points)

    summary = {
        "status": certificate.final_status,
        "run_dir": str(out_root),
        "spec_digest": certificate.spec_digest,
        "candidate_digest": certificate.candidate_digest,
        "gates": certificate.gates,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
