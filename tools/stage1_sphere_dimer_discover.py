#!/usr/bin/env python3
from __future__ import annotations

"""
Stage-1 sphere dimer discovery driver (Kelvin ladder + rings).

This is a small wrapper around electrodrive.tools.images_discover to keep
canonical parameters for the Stage-1 inside-spec smoke runs.
"""

import argparse
import json
from pathlib import Path
from typing import List

from electrodrive.tools.images_discover import _load_spec, _parse_basis_arg
from electrodrive.images.search import discover_images
from electrodrive.images.io import save_image_system
from electrodrive.orchestration.spec_registry import stage1_sphere_dimer_inside_path
from electrodrive.utils.logging import JsonlLogger


DEFAULT_SPEC = stage1_sphere_dimer_inside_path()
DEFAULT_BASIS = "sphere_kelvin_ladder,sphere_equatorial_ring"


def run_discover(
    spec_path: Path,
    basis: List[str],
    nmax: int,
    reg_l1: float,
    restarts: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)
    logger.info(
        "Stage1 sphere dimer discovery started.",
        spec=str(spec_path),
        basis=basis,
        nmax=int(nmax),
        reg_l1=float(reg_l1),
        restarts=int(restarts),
    )
    spec = _load_spec(spec_path)
    system = discover_images(
        spec=spec,
        basis_types=basis,
        n_max=nmax,
        reg_l1=reg_l1,
        restarts=restarts,
        logger=logger,
    )
    save_image_system(
        system,
        out_dir / "discovered_system.json",
        metadata={
            "spec_path": str(spec_path.resolve()),
            "basis_types": basis,
            "n_max": int(nmax),
            "reg_l1": float(reg_l1),
            "restarts": int(restarts),
        },
    )
    logger.info("Stage1 discovery completed.", n_images=len(system.elements))
    logger.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 sphere dimer discovery (Kelvin ladder + rings).")
    p.add_argument("--config", type=Path, help="Optional JSON config overriding defaults.")
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Spec path (default inside-lens).")
    p.add_argument("--basis", type=str, default=DEFAULT_BASIS, help="Comma-separated basis types.")
    p.add_argument("--nmax", type=int, default=8, help="Max images after sparsification.")
    p.add_argument("--reg-l1", type=float, default=1e-3, help="L1 regularisation.")
    p.add_argument("--restarts", type=int, default=1, help="Number of LBFGS restarts.")
    p.add_argument("--out", type=Path, default=Path("runs/stage1_sphere_dimer/discover_smoke"), help="Output directory.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        spec = Path(cfg.get("spec", args.spec))
        basis = _parse_basis_arg(cfg.get("basis", DEFAULT_BASIS))
        nmax = int(cfg.get("nmax", args.nmax))
        reg_l1 = float(cfg.get("reg_l1", args.reg_l1))
        restarts = int(cfg.get("restarts", args.restarts))
        out_dir = Path(cfg.get("out", args.out))
    else:
        spec = args.spec
        basis = _parse_basis_arg(args.basis)
        nmax = args.nmax
        reg_l1 = args.reg_l1
        restarts = args.restarts
        out_dir = args.out

    run_discover(spec, basis, nmax, reg_l1, restarts, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
