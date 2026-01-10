#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml

from electrodrive.images.basis import ImageBasisElement
from electrodrive.images.search import ImageSystem
from electrodrive.learn.collocation import compute_layered_reference_potential
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.verify.oracle_types import OracleFidelity
from electrodrive.verify.verifier import VerificationPlan, Verifier


@dataclass(frozen=True)
class CandidatePaths:
    summary_path: Path
    cert_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _oracle_for_name(name: str) -> OracleFidelity:
    key = (name or "fast").strip().lower()
    if key in {"mid", "f1"}:
        return OracleFidelity.F1
    if key in {"hi", "f2"}:
        return OracleFidelity.F2
    return OracleFidelity.F0


def _deserialize_element(entry: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> ImageBasisElement:
    elem_type = str(entry.get("type", ""))
    if elem_type.startswith("dcim_block"):
        from electrodrive.images.basis_dcim import DCIMBlockBasis

        return DCIMBlockBasis.deserialize(entry)
    return ImageBasisElement.deserialize(entry, device=device, dtype=dtype)


def _extract_spec(cert: Dict[str, Any]) -> Dict[str, Any]:
    gates = cert.get("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("certificate missing gates")
    for gate in ("A", "B", "C", "D"):
        config = gates.get(gate, {}).get("config", {})
        spec = config.get("spec")
        if isinstance(spec, dict) and spec:
            return spec
    raise ValueError("certificate missing spec in gate config")


def _parse_pairs_from_file(path: Path) -> List[CandidatePaths]:
    pairs: List[CandidatePaths] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"expected summary+cert paths on line: {raw}")
        pairs.append(CandidatePaths(Path(parts[0]), Path(parts[1])))
    return pairs


def _parse_pairs_from_args(pairs: Optional[Sequence[Sequence[str]]]) -> List[CandidatePaths]:
    if not pairs:
        return []
    out = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"expected summary+cert paths, got {pair}")
        out.append(CandidatePaths(Path(pair[0]), Path(pair[1])))
    return out


def _resolve_run_dir(summary_path: Path) -> Path:
    if len(summary_path.parents) < 3:
        raise ValueError(f"summary path too shallow: {summary_path}")
    return summary_path.parents[2]


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = _load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        return None
    return cfg


def _use_reference_potential(cfg: Optional[Dict[str, Any]]) -> bool:
    if not cfg:
        return False
    run_cfg = cfg.get("run", {})
    if not isinstance(run_cfg, dict):
        return False
    return bool(run_cfg.get("use_reference_potential", False))


def _base_plan_from_config(cfg: Optional[Dict[str, Any]]) -> VerificationPlan:
    plan = VerificationPlan()
    if cfg:
        oracle_cfg = cfg.get("oracle", {})
        if isinstance(oracle_cfg, dict):
            fast_cfg = oracle_cfg.get("fast", {})
            if isinstance(fast_cfg, dict):
                plan.start_fidelity = _oracle_for_name(str(fast_cfg.get("name", "fast")))
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, dict):
            plan.allow_f2 = bool(run_cfg.get("use_hi_oracle", False))
    plan.oracle_budget = dict(plan.oracle_budget)
    plan.oracle_budget["allow_cpu_fallback"] = False
    return plan


def _plan_for_seed(base: VerificationPlan, seed: int) -> VerificationPlan:
    plan = VerificationPlan()
    plan.gate_order = list(base.gate_order)
    plan.samples = dict(base.samples)
    plan.thresholds = dict(base.thresholds)
    plan.oracle_budget = dict(base.oracle_budget)
    plan.allow_f2 = base.allow_f2
    plan.allow_f3 = base.allow_f3
    plan.start_fidelity = base.start_fidelity
    plan.artifact_verbosity = base.artifact_verbosity
    plan.seeds = {k: int(v) + int(seed) for k, v in base.seeds.items()}
    return plan


def _candidate_tag(summary_path: Path) -> str:
    stem = summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    run_name = _resolve_run_dir(summary_path).name
    return f"{run_name}_{stem}"


def _gate_statuses(cert_json: Dict[str, Any]) -> Dict[str, str]:
    gates = cert_json.get("gates", {}) if isinstance(cert_json, dict) else {}
    statuses: Dict[str, str] = {}
    for gate in ("A", "B", "C", "D", "E"):
        statuses[gate] = str(gates.get(gate, {}).get("status", "missing"))
    return statuses


def _gate_metrics(cert_json: Dict[str, Any], gate: str) -> Dict[str, Any]:
    gates = cert_json.get("gates", {}) if isinstance(cert_json, dict) else {}
    metrics = gates.get(gate, {}).get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _is_success(statuses: Dict[str, str]) -> bool:
    if not all(statuses.get(g) == "pass" for g in ("A", "B", "C", "D")):
        return False
    return statuses.get("E") != "fail"


def _summarize_metrics(cert_json: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {
        "A.linf": _gate_metrics(cert_json, "A").get("linf"),
        "A.p95": _gate_metrics(cert_json, "A").get("p95"),
        "B.dirichlet_max_err": _gate_metrics(cert_json, "B").get("dirichlet_max_err"),
        "B.interface_max_v_jump": _gate_metrics(cert_json, "B").get("interface_max_v_jump"),
        "B.interface_max_d_jump": _gate_metrics(cert_json, "B").get("interface_max_d_jump"),
        "C.far_slope": _gate_metrics(cert_json, "C").get("far_slope"),
        "C.near_slope": _gate_metrics(cert_json, "C").get("near_slope"),
        "C.spurious_fraction": _gate_metrics(cert_json, "C").get("spurious_fraction"),
        "D.relative_change": _gate_metrics(cert_json, "D").get("relative_change"),
        "E.speedup": _gate_metrics(cert_json, "E").get("speedup"),
    }
    return {k: v for k, v in metrics.items() if v is not None}


def reverify_candidates(
    pairs: Sequence[CandidatePaths],
    out_root: Path,
    seeds: Sequence[int],
    max_candidates: int,
) -> Optional[Path]:
    device = torch.device("cuda")
    dtype = torch.float32
    verifier = Verifier(out_root=out_root)

    cfg_cache: Dict[Path, Optional[Dict[str, Any]]] = {}
    plan_cache: Dict[Path, VerificationPlan] = {}

    for idx, pair in enumerate(pairs[:max_candidates], start=1):
        summary_path = pair.summary_path
        cert_path = pair.cert_path
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary: {summary_path}")
        if not cert_path.exists():
            raise FileNotFoundError(f"missing certificate: {cert_path}")

        summary = _load_json(summary_path)
        cert = _load_json(cert_path)
        spec_dict = _extract_spec(cert)
        spec_obj = CanonicalSpec.from_json(spec_dict)

        run_dir = _resolve_run_dir(summary_path)
        if run_dir not in cfg_cache:
            cfg_cache[run_dir] = _load_run_config(run_dir)
        cfg = cfg_cache[run_dir]

        if run_dir not in plan_cache:
            plan_cache[run_dir] = _base_plan_from_config(cfg)
        base_plan = plan_cache[run_dir]
        use_ref = _use_reference_potential(cfg)

        elements = [
            _deserialize_element(entry, device=device, dtype=dtype)
            for entry in summary.get("elements", [])
        ]
        weights = torch.tensor(summary.get("weights", []), device=device, dtype=dtype)
        system = ImageSystem(elements, weights)

        def candidate_eval(pts: torch.Tensor) -> torch.Tensor:
            if not pts.is_cuda:
                raise ValueError("candidate_eval requires CUDA points")
            out = system.potential(pts)
            if use_ref:
                out = out + compute_layered_reference_potential(
                    spec_obj,
                    pts,
                    device=pts.device,
                    dtype=pts.dtype,
                )
            return out

        tag = _candidate_tag(summary_path)
        print(f"[{idx:02d}] candidate={tag} use_reference_potential={use_ref}")

        seed_results: List[Tuple[int, Path, Dict[str, str], Dict[str, Any]]] = []
        candidate_ok = True
        for seed in seeds:
            plan = _plan_for_seed(base_plan, seed)
            out_dir = out_root / tag / f"seed{seed}"
            if out_dir.exists():
                raise FileExistsError(f"refusing to overwrite existing run dir: {out_dir}")
            certificate = verifier.run({"eval_fn": candidate_eval}, spec_dict, plan, outdir=out_dir)
            cert_json = certificate.to_json()
            statuses = _gate_statuses(cert_json)
            metrics = _summarize_metrics(cert_json)
            seed_results.append((seed, out_dir, statuses, metrics))
            status_line = " ".join([f"{g}={statuses.get(g)}" for g in ("A", "B", "C", "D", "E")])
            print(f"  seed={seed} status {status_line}")
            if not _is_success(statuses):
                candidate_ok = False
                break

        if candidate_ok:
            print("SUCCESS candidate met stability requirements:")
            for seed, out_dir, statuses, metrics in seed_results:
                status_line = " ".join([f"{g}={statuses.get(g)}" for g in ("A", "B", "C", "D", "E")])
                print(f"  seed={seed} cert={out_dir / 'discovery_certificate.json'} {status_line}")
                if metrics:
                    metrics_line = " ".join([f"{k}={v}" for k, v in metrics.items()])
                    print(f"    metrics {metrics_line}")
            return seed_results[-1][1] / "discovery_certificate.json"

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="text file with summary_path,cert_path per line")
    parser.add_argument("--pair", nargs=2, action="append", help="summary_path cert_path", default=[])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    pairs = []
    if args.input:
        pairs.extend(_parse_pairs_from_file(Path(args.input)))
    pairs.extend(_parse_pairs_from_args(args.pair))
    if not pairs:
        raise SystemExit("No candidate pairs provided.")

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seed_list:
        raise SystemExit("No seeds provided.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cert_path = reverify_candidates(
        pairs=pairs,
        out_root=out_root,
        seeds=seed_list,
        max_candidates=args.max_candidates,
    )
    if cert_path:
        print(f"SUCCESS certificate: {cert_path}")
    else:
        print("No SUCCESS candidate found.")


if __name__ == "__main__":
    main()
