#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.verify.oracle_backends import F1SommerfeldOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, OracleFidelity, OracleQuantity, OracleQuery
from electrodrive.verify.utils import normalize_dtype
from electrodrive.verify.verifier import VerificationPlan, Verifier


def _make_three_layer_spec(eps2: float, slab_thickness: float, z_charge: float) -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": eps2, "z_min": -slab_thickness, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -slab_thickness},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, z_charge]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def _env_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "",
    }
    if torch.cuda.is_available():
        info.update(
            {
                "device": "cuda",
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return info


def _git_report() -> Dict[str, str]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    return {"sha": sha, "branch": branch}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate verifier Gate A on a layered F1 oracle.")
    parser.add_argument("--out-root", default="runs/black_hammer_calibration", help="Output root directory.")
    parser.add_argument("--eps2", type=float, default=4.0, help="Slab dielectric constant.")
    parser.add_argument("--slab-thickness", type=float, default=0.3, help="Slab thickness.")
    parser.add_argument("--z-charge", type=float, default=0.2, help="Charge height above interface.")
    parser.add_argument("--n-interior", type=int, default=64, help="Gate A interior samples.")
    parser.add_argument("--exclusion-radius", type=float, default=0.1, help="Gate A exclusion radius.")
    parser.add_argument("--linf-tol", type=float, default=5e-3, help="Gate A linf tolerance.")
    parser.add_argument("--fd-h", type=float, default=2e-2, help="Gate A finite-diff step.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for verifier calibration.")

    spec = _make_three_layer_spec(args.eps2, args.slab_thickness, args.z_charge)
    spec_dict = spec.to_json()

    backend = F1SommerfeldOracleBackend()

    def _candidate_eval(pts: torch.Tensor) -> torch.Tensor:
        query = OracleQuery(
            spec=spec_dict,
            points=pts,
            quantity=OracleQuantity.POTENTIAL,
            requested_fidelity=OracleFidelity.F1,
            device=str(pts.device),
            dtype=normalize_dtype(pts.dtype),
            cache_policy=CachePolicy.OFF,
            budget={},
        )
        result = backend.evaluate(query)
        if result.V is None:
            return torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)
        return result.V

    plan = VerificationPlan()
    plan.gate_order = ["A"]
    plan.start_fidelity = OracleFidelity.F1
    plan.samples["A_interior"] = int(args.n_interior)
    plan.thresholds["laplacian_linf"] = float(args.linf_tol)
    plan.thresholds["laplacian_exclusion_radius"] = float(args.exclusion_radius)
    plan.thresholds["laplacian_fd_h"] = float(args.fd_h)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan.run_name = stamp

    verifier = Verifier(out_root=out_root)
    cert = verifier.run({"eval_fn": _candidate_eval}, spec_dict, plan)

    run_dir = out_root / stamp
    (run_dir / "env.json").write_text(json.dumps(_env_report(), indent=2), encoding="utf-8")
    (run_dir / "git.json").write_text(json.dumps(_git_report(), indent=2), encoding="utf-8")
    (run_dir / "plan.json").write_text(json.dumps(plan.to_json(), indent=2), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec_dict, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(cert.to_json(), indent=2), encoding="utf-8")

    print(f"Calibration run dir: {run_dir}")
    print(f"Final status: {cert.final_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
