#!/usr/bin/env python3
from __future__ import annotations

"""
Stage-1 double-sphere (sphere dimer) BEM probe harness.

Purpose
-------
- Load canonical Stage-1 specs (two grounded spheres on z-axis + axis charge).
- Run a small ladder of BEM attempts (GPU/CPU, fp64/fp32) with near-field quadrature enabled.
- Classify numeric health (OK/WARN/FAIL) using the same logic as _bem_probe.py.
- Emit compact summary JSON under runs/stage1_sphere_dimer/ for later steps.

Usage
-----
    PYTHONPATH=. python tools/stage1_sphere_dimer_bem_probe.py \
        --spec specs/stage1_sphere_dimer_axis_point_inside.json \
        --out runs/stage1_sphere_dimer/probe_inside
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from electrodrive.utils.config import BEMConfig
from electrodrive.utils.logging import JsonlLogger
from electrodrive.orchestration.spec_registry import stage1_sphere_dimer_inside_path

from _bem_probe import (  # reuse existing probe helpers
    DictSpecWrapper,
    evaluate_numeric_health,
    run_bem_attempt,
    try_load_spec,
)


DEFAULT_SPECS = [
    stage1_sphere_dimer_inside_path(),
    Path("specs/stage1_sphere_dimer_axis_point_left.json"),
    Path("specs/stage1_sphere_dimer_axis_point_right.json"),
]


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_spec(path: Path) -> Any:
    spec = try_load_spec(path)
    if isinstance(spec, dict):
        return DictSpecWrapper(spec)
    return spec


def make_base_cfg() -> BEMConfig:
    cfg = BEMConfig()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.fp64 = True
    cfg.initial_h = 0.25
    cfg.max_refine_passes = 3
    cfg.use_near_quadrature = True
    cfg.use_near_quadrature_matvec = True
    cfg.near_quadrature_order = 2
    cfg.near_quadrature_distance_factor = 2.0
    cfg.vram_autotune = False
    cfg.gmres_tol = 1e-8
    return cfg


def make_oracle_cfg() -> BEMConfig:
    cfg = make_base_cfg()
    cfg.initial_h = 0.2
    cfg.max_refine_passes = 4
    cfg.gmres_tol = 1e-9
    return cfg


@dataclass
class ProbeSummary:
    spec_path: str
    status: str
    reasons: List[str]
    attempts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_path": self.spec_path,
            "status": self.status,
            "reasons": self.reasons,
            "attempts": self.attempts,
        }


def _attempt_ladder() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("oracle", {"initial_h": 0.2, "max_refine_passes": 4, "gmres_tol": 1e-9}),
        ("base_gpu_fp64", {}),
        ("cpu_fp64", {"use_gpu": False}),
        ("gpu_fp32", {"fp64": False}),
        ("cpu_fp32", {"use_gpu": False, "fp64": False}),
        ("coarse", {"max_refine_passes": 2, "initial_h": 0.3}),
    ]


def probe_spec(
    spec_path: Path,
    out_root: Path,
    *,
    attempts: Sequence[Tuple[str, Dict[str, Any]]] | None = None,
    mode: str = "ladder",
    backend: str = "bem",
) -> ProbeSummary:
    if backend.lower() != "bem":
        raise NotImplementedError("FMM/backend!=bem plumbing reserved for later Stage 1 steps.")
    set_seeds(1234)
    spec_for_bem = _load_spec(spec_path)
    base_cfg = make_oracle_cfg() if mode == "oracle" else make_base_cfg()

    attempt_logs: List[Dict[str, Any]] = []
    best_status = "fail"
    best_reasons: List[str] = ["no_attempts"]
    order = {"ok": 0, "warn": 1, "fail": 2}

    if attempts is not None:
        ladder = list(attempts)
    elif mode == "oracle":
        ladder = [("oracle", {"initial_h": 0.2, "max_refine_passes": 4, "gmres_tol": 1e-9})]
    else:
        ladder = _attempt_ladder()

    for label, overrides in ladder:
        attempt_dir = out_root / label
        os.environ["EDE_RUN_DIR"] = str(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        with JsonlLogger(attempt_dir) as logger:
            res = run_bem_attempt(spec_for_bem, base_cfg, logger, label, overrides)

        mesh_stats = (res.out_dict or {}).get("mesh_stats", {}) if res.out_dict else {}
        gmres_stats = (res.out_dict or {}).get("gmres_stats", {}) if res.out_dict else {}
        attempt_logs.append(
            {
                "label": label,
                "health_status": res.health_status,
                "health_reasons": res.health_reasons,
                "error_type": res.error_type,
                "error_message": res.error_message,
                "mesh_stats": mesh_stats,
                "gmres_stats": gmres_stats,
                "cfg_used": res.cfg_used,
            }
        )

        status = res.health_status or "fail"
        if order[status] <= order[best_status]:
            best_status = status
            best_reasons = res.health_reasons
        if best_status == "ok":
            break

    summary = ProbeSummary(
        spec_path=str(spec_path),
        status=best_status,
        reasons=best_reasons,
        attempts=attempt_logs,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary


def run_all(specs: Sequence[Path], out_root: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for sp in specs:
        spec_out = out_root / sp.stem
        summary = probe_spec(sp, spec_out)
        results[str(sp)] = summary.to_dict()
        print(
            f"[Stage1 probe] {sp.name}: status={summary.status} reasons={summary.reasons}"
        )
    (out_root / "all_summaries.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 sphere dimer BEM probe harness.")
    p.add_argument("--spec", action="append", type=Path, help="Spec path(s) to probe. Default: canonical Stage-1 dimer specs.")
    p.add_argument("--out", type=Path, default=Path("runs/stage1_sphere_dimer/bem_probe"), help="Output root directory.")
    p.add_argument("--mode", choices=["ladder", "oracle"], default="ladder", help="Probe mode: full ladder or oracle-only attempt.")
    p.add_argument("--backend", choices=["bem", "fmm"], default="bem", help="Reserved for future FMM support.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = args.spec if args.spec else DEFAULT_SPECS
    if args.mode == "oracle":
        results = {}
        for sp in specs:
            spec_out = args.out / "oracle" / sp.stem
            summary = probe_spec(sp, spec_out, mode="oracle", backend=args.backend)
            results[str(sp)] = summary.to_dict()
            print(f"[Stage1 probe][oracle] {sp.name}: status={summary.status} reasons={summary.reasons}")
        (args.out / "oracle_all_summaries.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        run_all(specs, args.out)


if __name__ == "__main__":
    main()
