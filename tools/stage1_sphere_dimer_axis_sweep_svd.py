#!/usr/bin/env python3
from __future__ import annotations

"""
Axis sweep + SVD for Stage-1 sphere dimer.

For a fixed geometry and basis configuration, sweep source z positions,
run discovery, collect weight vectors, and compute the SVD of the weight matrix.
"""

import argparse
import json
import datetime
import shutil
from pathlib import Path
from typing import Any, List, Sequence, Optional, Tuple, Dict

import numpy as np
import torch

from electrodrive.images.search import discover_images
from electrodrive.images.io import save_image_system
from electrodrive.images.weight_modes import (
    compute_weight_modes,
    export_weight_mode_bundle,
    fit_symbolic_modes,
    fit_quality_ok,
    load_symbolic_fits,
    load_svd_bundle,
    predict_weights_from_modes,
    render_summary,
    spectral_gap_ok,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.orchestration.spec_registry import stage1_sphere_dimer_inside_path
from electrodrive.utils.logging import JsonlLogger

from tools.stage1_sphere_dimer_discover import _parse_basis_arg  # type: ignore


DEFAULT_SPEC = stage1_sphere_dimer_inside_path()
DEFAULT_ZS = [0.3, 0.7, 1.0, 1.3, 1.6, 1.9, 2.1]


def load_spec(path: Path) -> CanonicalSpec:
    return CanonicalSpec.from_json(json.loads(path.read_text()))


def set_charge_z(spec: CanonicalSpec, z: float) -> CanonicalSpec:
    data = {
        "domain": spec.domain,
        "BCs": spec.BCs,
        "conductors": spec.conductors,
        "charges": [],
        "symmetry": spec.symmetry,
        "domain_meta": getattr(spec, "domain_meta", {}),
    }
    ch = spec.charges[0]
    ch_new = dict(ch)
    ch_new["pos"] = [0.0, 0.0, float(z)]
    data["charges"] = [ch_new]
    return CanonicalSpec.from_json(data)


def run_for_z(
    spec: CanonicalSpec,
    basis: List[str],
    nmax: int,
    reg_l1: float,
    restarts: int,
    out_dir: Path,
    weight_prior: Optional[np.ndarray] = None,
    lambda_weight_prior: float = 0.0,
) -> Any:
    logger = JsonlLogger(out_dir)
    system = discover_images(
        spec,
        basis_types=basis,
        n_max=nmax,
        reg_l1=reg_l1,
        restarts=restarts,
        logger=logger,
        weight_prior=weight_prior,
        lambda_weight_prior=lambda_weight_prior,
        weight_prior_label="weight_modes" if weight_prior is not None else None,
    )
    save_image_system(system, out_dir / "discovered_system.json", metadata={"z": spec.charges[0]["pos"][2]})
    logger.close()
    return system


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Axis sweep + SVD + symbolic fits for Stage-1 sphere dimer.")
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--basis", type=str, default="sphere_kelvin_ladder,sphere_equatorial_ring")
    p.add_argument("--nmax", type=int, default=8)
    p.add_argument("--reg-l1", type=float, default=1e-3)
    p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--z", nargs="+", type=float, default=DEFAULT_ZS)
    p.add_argument("--out", type=Path, default=Path("runs/stage1_sphere_dimer/axis_sweep_svd"))
    p.add_argument("--max-rank", type=int, default=3, help="Top singular modes to keep for fits/prior.")
    p.add_argument("--max-poly-degree", type=int, default=4, help="Max polynomial degree for symbolic fits.")
    p.add_argument("--use-weight-modes", action="store_true", help="Enable mode-controller prior if fits are provided.")
    p.add_argument("--mode-dir", type=Path, help="Directory containing svd_modes.npy and symbolic_fits.json for the controller.")
    p.add_argument("--lambda-weight-mode", type=float, default=0.0, help="Quadratic prior strength for weight-mode controller.")
    p.add_argument("--spectral-gap-thresh", type=float, default=0.1, help="Spectral gap threshold to trust mode controller.")
    p.add_argument("--rel-rmse-thresh", type=float, default=0.2, help="Max relative RMSE allowed for fitted modes.")
    p.add_argument("--vault", action="store_true", help="Copy artifacts into the_vault for audit.")
    p.add_argument("--vault-slug", type=str, help="Optional vault subdir name (otherwise timestamped).")
    args = p.parse_args(argv)

    basis = _parse_basis_arg(args.basis)
    spec0 = load_spec(args.spec)
    zs: Sequence[float] = args.z

    controller_bundle: Optional[Dict[str, Any]] = None
    controller_fits: List[Dict[str, Any]] = []
    if args.use_weight_modes and args.mode_dir:
        svd_path = args.mode_dir / "svd_modes.npy"
        fits_path = args.mode_dir / "symbolic_fits.json"
        bundle = load_svd_bundle(svd_path)
        _, fits_loaded = load_symbolic_fits(fits_path)
        if bundle and fits_loaded:
            if spectral_gap_ok(bundle.get("S", []), rank=args.max_rank, thresh=args.spectral_gap_thresh) and fit_quality_ok(
                fits_loaded, rel_rmse_tol=args.rel_rmse_thresh
            ):
                controller_bundle = bundle
                controller_fits = fits_loaded
                print(f"[controller] Loaded prior from {args.mode_dir}")
            else:
                print("[controller] Skipping controller due to spectral gap or fit quality.")

    weights: List[torch.Tensor] = []
    systems = []
    for z in zs:
        spec_z = set_charge_z(spec0, z)
        run_dir = args.out / f"z_{str(z).replace('.', 'p')}"
        w_prior = None
        if controller_bundle is not None and controller_fits:
            w_prior = predict_weights_from_modes(z, controller_bundle, controller_fits, max_rank=args.max_rank)
        system = run_for_z(
            spec_z,
            basis,
            args.nmax,
            args.reg_l1,
            args.restarts,
            run_dir,
            weight_prior=w_prior,
            lambda_weight_prior=args.lambda_weight_mode,
        )
        weights.append(system.weights.detach().cpu())
        systems.append(system)

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = compute_weight_modes(weights, zs, max_rank=args.max_rank)
    fits = fit_symbolic_modes(zs, bundle.mode_curves, max_rank=args.max_rank, max_poly_degree=args.max_poly_degree)
    export_weight_mode_bundle(
        args.out,
        bundle,
        fits,
        extra_metrics={
            "spec": str(args.spec),
            "basis": basis,
            "nmax": args.nmax,
            "reg_l1": args.reg_l1,
            "restarts": args.restarts,
        },
    )
    np.savez(args.out / "weights_svd.npz", W=bundle.weights, U=bundle.U, S=bundle.S, VT=bundle.VT, z=zs)
    summary = {
        "spec": str(args.spec),
        "basis": basis,
        "nmax": args.nmax,
        "reg_l1": args.reg_l1,
        "restarts": args.restarts,
        "z_grid": list(zs),
        "singular_values": bundle.S.tolist(),
        "sigma_norm": bundle.sigma_norm.tolist(),
        "effective_rank": bundle.effective_rank,
        "controller_used": bool(controller_bundle is not None and controller_fits),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    geometry_desc = f"Stage-1 sphere dimer (spec={args.spec.name})"
    research_wishlist = [
        "Connect leading mode to Kelvin ladder for symmetric dimers; derive decay rate of singular values from symmetry.",
        "Relate rational fit poles to effective image distances; test against analytic two-sphere Neumann series.",
        "Extend controller beyond axis by coupling to off-axis collocation or low-order spherical harmonics.",
    ]
    summary_text = render_summary(
        label="Stage-1 axis sweep weight modes",
        geometry=geometry_desc,
        basis=basis,
        z_grid=zs,
        bundle=bundle,
        fits=fits,
        research_wishlist=research_wishlist,
    )
    (args.out / "summary.md").write_text(summary_text, encoding="utf-8")
    print("SVD summary:", summary)

    if args.vault or args.vault_slug:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = args.vault_slug or f"stage1_axis_weight_modes_{ts}"
        vault_dir = Path("the_vault") / slug
        vault_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["weights_vs_axis.npy", "svd_modes.npy", "symbolic_fits.json", "metrics.json", "summary.md", "summary.json"]:
            src = args.out / fname
            if src.exists():
                shutil.copy(src, vault_dir / fname)
        try:
            shutil.copy(args.spec, vault_dir / args.spec.name)
        except Exception:
            pass
        print(f"Vaulted artifacts to {vault_dir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
