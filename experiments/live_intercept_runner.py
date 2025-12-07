#!/usr/bin/env python
"""
Run the single plane analytic-vs-BEM test with live interception enabled,
then print a concise summary of the captured metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


def load_log(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(payload: Dict[str, Any]) -> None:
    print(f"[summary] run_id={payload.get('run_id')} mode={payload.get('mode')} geom={payload.get('geom')}")
    passes = payload.get("refinement_passes", [])
    if not passes:
        print("[summary] no refinement passes recorded.")
    else:
        for p in passes:
            print(
                f"[pass {p.get('pass')}] h={p.get('h')} n_panels={p.get('n_panels')} "
                f"bc_resid_linf={p.get('bc_resid_linf')} gmres_iters={p.get('gmres_iters')} "
                f"gmres_resid_true={p.get('gmres_resid_true')} tol_abs={p.get('gmres_tol_abs')}"
            )
    colloc = payload.get("collocation", {})
    if colloc:
        print(
            "[collocation] n_points={n} ratio_boundary={rb} rel_err_overall_max={o} "
            "rel_err_boundary_max={b} rel_err_interior_max={i}".format(
                n=colloc.get("n_points"),
                rb=colloc.get("ratio_boundary"),
                o=colloc.get("rel_err_overall_max"),
                b=colloc.get("rel_err_boundary_max"),
                i=colloc.get("rel_err_interior_max"),
            )
        )
        if "samples" in colloc:
            worst = max(colloc["samples"], key=lambda s: abs(s.get("rel_err", 0.0)), default=None)
            if worst:
                print(
                    f"[collocation] worst sample err={worst.get('rel_err')} "
                    f"point={worst.get('point')} boundary={worst.get('is_boundary')}"
                )
    else:
        print("[collocation] not recorded.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["short", "deep"], default="short")
    parser.add_argument("--geom", default="plane")
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--outdir", default="experiments/_agent_outputs")
    parser.add_argument("--skip-if-existing", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.geom}_{args.mode}_{int(time.time())}"
    outdir = Path(args.outdir)
    out_path = outdir / f"{run_id}.json"

    if args.skip_if_existing and out_path.is_file():
        payload = load_log(out_path)
        print_summary(payload)
        return 0

    env = os.environ.copy()
    env["EDE_BEM_INTERCEPT_MODE"] = args.mode
    env["EDE_BEM_INTERCEPT_GEOM"] = args.geom
    env["EDE_BEM_INTERCEPT_OUTDIR"] = str(outdir)
    env["EDE_BEM_INTERCEPT_RUN_ID"] = run_id
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_bem_quadrature.py::test_analytic_matches_bem_up_to_units[_build_plane_spec-plane-True]",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    print(f"[runner] running pytest with mode={args.mode} run_id={run_id}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"[runner] pytest exited with code {res.returncode}", file=sys.stderr)

    if not out_path.is_file():
        print(f"[runner] log not found at {out_path}", file=sys.stderr)
        return res.returncode or 1

    payload = load_log(out_path)
    print_summary(payload)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
