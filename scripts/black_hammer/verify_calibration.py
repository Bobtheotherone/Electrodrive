#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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


def _parse_gate_order(text: str) -> list[str]:
    gates = [gate.strip().upper() for gate in text.split(",") if gate.strip()]
    return gates or ["A"]


def _worst_metric(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    worst_key = None
    worst_ratio = None
    worst_value = None
    worst_threshold = None
    for key, value in metrics.items():
        if key not in thresholds:
            continue
        try:
            threshold = float(thresholds[key])
            if threshold == 0.0:
                ratio = float("inf")
            else:
                ratio = float(value) / threshold
        except (TypeError, ValueError):
            continue
        if worst_ratio is None or ratio > worst_ratio:
            worst_ratio = ratio
            worst_key = key
            worst_value = float(value)
            worst_threshold = float(thresholds[key])
    return worst_key, worst_ratio, worst_value, worst_threshold


def _summarize_certificate(cert: Any) -> None:
    cert_json = cert.to_json()
    print(f"final_status: {cert_json.get('final_status')}")
    print(f"gate_order: {cert_json.get('gate_order')}")
    gates = cert_json.get("gates") or {}
    print("gates:")
    for gate_name, gate in gates.items():
        if not isinstance(gate, dict):
            continue
        status = gate.get("status")
        metrics = gate.get("metrics") or {}
        thresholds = gate.get("thresholds") or {}
        worst_key, worst_ratio, worst_val, worst_thr = _worst_metric(metrics, thresholds)
        line = f"- {gate_name}: status={status}"
        if worst_key is not None and worst_ratio is not None:
            line += f" worst={worst_key} ratio={worst_ratio:.3e} value={worst_val} threshold={worst_thr}"
        if metrics:
            line += f" metrics={metrics}"
        if thresholds:
            line += f" thresholds={thresholds}"
        print(line)


def _plan_thresholds_payload(plan: VerificationPlan) -> str:
    return json.dumps({k: float(v) for k, v in plan.thresholds.items()}, sort_keys=True)


def _write_outputs(run_dir: Path, plan: VerificationPlan, spec_dict: Dict[str, Any], cert: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(_env_report(), indent=2), encoding="utf-8")
    (run_dir / "git.json").write_text(json.dumps(_git_report(), indent=2), encoding="utf-8")
    (run_dir / "plan.json").write_text(json.dumps(plan.to_json(), indent=2), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec_dict, indent=2), encoding="utf-8")
    cert_json = cert.to_json()
    (run_dir / "discovery_certificate.json").write_text(json.dumps(cert_json, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(cert_json, indent=2), encoding="utf-8")


def _laplacian_configs() -> Iterable[Tuple[float, float, int]]:
    for exclusion_radius in (0.05, 0.1, 0.2, 0.4):
        for fd_h in (0.02, 0.01, 0.005):
            yield exclusion_radius, fd_h, 1


def _build_plan(
    gate_order: list[str],
    n_interior: int,
    linf_tol: float,
    exclusion_radius: float,
    fd_h: float,
    prefer_autograd: int,
) -> VerificationPlan:
    plan = VerificationPlan()
    plan.gate_order = list(gate_order)
    plan.start_fidelity = OracleFidelity.F1
    plan.samples["A_interior"] = int(n_interior)
    plan.thresholds["laplacian_linf"] = float(linf_tol)
    plan.thresholds["laplacian_exclusion_radius"] = float(exclusion_radius)
    plan.thresholds["laplacian_fd_h"] = float(fd_h)
    plan.thresholds["laplacian_prefer_autograd"] = float(prefer_autograd)
    return plan


def _candidate_eval_factory(
    candidate: str,
    spec_dict: Dict[str, Any],
    backend: F1SommerfeldOracleBackend,
) -> Any:
    if candidate == "constant":
        return {"constant": 0.0}
    if candidate == "linear":
        def _linear_eval(pts: torch.Tensor) -> torch.Tensor:
            return pts[:, 2]

        return {"eval_fn": _linear_eval}

    def _f1_eval(pts: torch.Tensor) -> torch.Tensor:
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

    return {"eval_fn": _f1_eval}


def _gate_worst_ratio(cert: Any, gate_name: str) -> Optional[float]:
    cert_json = cert.to_json()
    gates = cert_json.get("gates") or {}
    gate = gates.get(gate_name)
    if not isinstance(gate, dict):
        return None
    metrics = gate.get("metrics") or {}
    thresholds = gate.get("thresholds") or {}
    _, ratio, _, _ = _worst_metric(metrics, thresholds)
    return ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate verifier Gate A on a layered F1 oracle.")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output directory (default: runs/black_hammer_calibration/<timestamp>/).",
    )
    parser.add_argument("--gate-order", default="A", help="Comma-separated gate order.")
    parser.add_argument("--candidate", choices=("constant", "linear", "f1"), default="constant", help="Candidate function.")
    parser.add_argument("--eps2", type=float, default=4.0, help="Slab dielectric constant.")
    parser.add_argument("--slab-thickness", type=float, default=0.3, help="Slab thickness.")
    parser.add_argument("--z-charge", type=float, default=0.2, help="Charge height above interface.")
    parser.add_argument("--n-interior", type=int, default=64, help="Gate A interior samples.")
    parser.add_argument("--linf-tol", type=float, default=5e-3, help="Gate A linf tolerance.")
    parser.add_argument("--laplacian-exclusion-radius", type=float, default=0.1, help="Gate A exclusion radius.")
    parser.add_argument("--laplacian-fd-h", type=float, default=2e-2, help="Gate A finite-diff step.")
    parser.add_argument("--laplacian-prefer-autograd", type=int, default=1, help="Gate A prefer autograd (1/0).")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for verifier calibration.")

    spec = _make_three_layer_spec(args.eps2, args.slab_thickness, args.z_charge)
    spec_dict = spec.to_json()
    gate_order = _parse_gate_order(args.gate_order)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else Path("runs/black_hammer_calibration") / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    backend = F1SommerfeldOracleBackend()
    candidate = _candidate_eval_factory(args.candidate, spec_dict, backend)

    verifier = Verifier(out_root=out_root)

    if args.candidate == "f1" and "A" in gate_order:
        (out_root / "env.json").write_text(json.dumps(_env_report(), indent=2), encoding="utf-8")
        (out_root / "git.json").write_text(json.dumps(_git_report(), indent=2), encoding="utf-8")
        best_ratio = None
        best_config: Optional[Dict[str, Any]] = None
        best_cert: Optional[Any] = None
        passing_cert: Optional[Any] = None
        last_cert: Optional[Any] = None
        found = False
        for exclusion_radius, fd_h, prefer_autograd in _laplacian_configs():
            plan = _build_plan(
                gate_order,
                args.n_interior,
                args.linf_tol,
                exclusion_radius,
                fd_h,
                prefer_autograd,
            )
            run_dir = out_root / f"sweep_ex{exclusion_radius}_h{fd_h}".replace(".", "p")
            print(f"plan.thresholds: {_plan_thresholds_payload(plan)}")
            print(f"sweep_config: exclusion_radius={exclusion_radius} fd_h={fd_h} prefer_autograd={prefer_autograd}")
            cert = verifier.run(candidate, spec_dict, plan, outdir=run_dir)
            last_cert = cert
            _write_outputs(run_dir, plan, spec_dict, cert)
            _summarize_certificate(cert)
            ratio = _gate_worst_ratio(cert, "A")
            current_config = {
                "exclusion_radius": exclusion_radius,
                "fd_h": fd_h,
                "prefer_autograd": prefer_autograd,
                "worst_ratio": ratio,
                "run_dir": str(run_dir),
            }
            if ratio is not None and (best_ratio is None or ratio < best_ratio):
                best_ratio = ratio
                best_config = dict(current_config)
                best_cert = cert
            if cert.final_status == "pass":
                found = True
                passing_cert = cert
                print(f"FOUND PASSING CONFIG: {current_config}")
                (out_root / "passing_knobs.json").write_text(json.dumps(current_config, indent=2), encoding="utf-8")
                break
        if not found:
            print(f"NO PASS FOUND. BEST CONFIG: {best_config}")
            if best_config is not None:
                (out_root / "best_knobs.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
        if passing_cert is not None:
            (out_root / "discovery_certificate.json").write_text(
                json.dumps(passing_cert.to_json(), indent=2), encoding="utf-8"
            )
        elif best_cert is not None:
            (out_root / "discovery_certificate.json").write_text(
                json.dumps(best_cert.to_json(), indent=2), encoding="utf-8"
            )
        elif last_cert is not None:
            (out_root / "discovery_certificate.json").write_text(
                json.dumps(last_cert.to_json(), indent=2), encoding="utf-8"
            )
    else:
        plan = _build_plan(
            gate_order,
            args.n_interior,
            args.linf_tol,
            args.laplacian_exclusion_radius,
            args.laplacian_fd_h,
            args.laplacian_prefer_autograd,
        )
        plan.run_name = out_root.name
        print(f"plan.thresholds: {_plan_thresholds_payload(plan)}")
        cert = verifier.run(candidate, spec_dict, plan, outdir=out_root)
        _write_outputs(out_root, plan, spec_dict, cert)
        _summarize_certificate(cert)

    print(f"Calibration run dir: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
