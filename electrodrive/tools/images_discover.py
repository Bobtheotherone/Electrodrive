from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch  # noqa: F401

from electrodrive.utils.logging import JsonlLogger
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.images.search import AugLagrangeConfig, discover_images
from electrodrive.images.io import save_image_system
from electrodrive.images.geo_encoder import GeoEncoder, load_geo_components_from_checkpoint
from electrodrive.images.learned_solver import load_lista_from_checkpoint
from electrodrive.images.learned_generator import SimpleGeoEncoder, MLPBasisGenerator
from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig


def _load_spec(path: Path) -> CanonicalSpec:
    """Load a CanonicalSpec from a JSON file.

    The helper intentionally accepts UTF-8 with or without a BOM so that
    specs saved from different editors remain usable.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return CanonicalSpec.from_json(raw)


def _parse_basis_arg(basis: str) -> List[str]:
    items = [b.strip() for b in basis.split(",") if b.strip()]
    return items or ["point"]


def run_discover(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)
    solver_cli = args.solver
    solver_explicit = solver_cli is not None
    if solver_explicit:
        solver_choice = solver_cli
    elif args.model_checkpoint and not args.aug_boundary:
        solver_choice = "lista"
    else:
        solver_choice = "ista"

    operator_mode_flag = args.operator_mode  # None means defer to discover/env
    operator_mode_log = True if operator_mode_flag is None else bool(operator_mode_flag)

    subtract_physical = bool(getattr(args, "subtract_physical", False))

    intensive = bool(getattr(args, "intensive", False))
    if intensive:
        if args.n_points is None:
            args.n_points = 8192
        if args.ratio_boundary is None:
            args.ratio_boundary = 0.7
        if args.adaptive_collocation_rounds is None:
            args.adaptive_collocation_rounds = 2
        if args.restarts is None:
            args.restarts = 3
        os.environ["EDE_IMAGES_INTENSIVE"] = "1"
    if args.adaptive_collocation_rounds is None:
        args.adaptive_collocation_rounds = 1
    if args.restarts is None:
        args.restarts = 1

    lambda_group = args.lambda_group
    if solver_choice == "lista" and (lambda_group is None or lambda_group == 0.0):
        lambda_group = 1e-3

    logger.info(
        "Images discovery run started.",
        spec=str(args.spec),
        basis=args.basis,
        nmax=int(args.nmax),
        reg_l1=float(args.reg_l1),
        restarts=int(args.restarts),
        basis_generator=args.basis_generator,
        basis_generator_mode=args.basis_generator_mode,
        model_checkpoint=args.model_checkpoint,
        solver=solver_choice,
        solver_explicit=bool(solver_explicit),
        operator_mode=operator_mode_log,
        adaptive_collocation_rounds=int(args.adaptive_collocation_rounds),
        lambda_group=float(lambda_group),
        n_points=args.n_points if args.n_points is not None else None,
        ratio_boundary=args.ratio_boundary if args.ratio_boundary is not None else None,
        aug_boundary=bool(args.aug_boundary),
        subtract_physical=subtract_physical,
        intensive=bool(intensive),
    )
    try:
        spec = _load_spec(Path(args.spec))
        basis_types = _parse_basis_arg(args.basis)
        lista_model = None
        basis_generator = None
        geo_encoder = None

        if args.model_checkpoint:
            lista_model = load_lista_from_checkpoint(args.model_checkpoint)
            geo_encoder, basis_generator = load_geo_components_from_checkpoint(args.model_checkpoint)
            if any([lista_model, geo_encoder, basis_generator]):
                logger.info(
                    "Model checkpoint hydrated.",
                    checkpoint=str(args.model_checkpoint),
                    has_lista=bool(lista_model),
                    has_geo_encoder=bool(geo_encoder),
                    has_basis_generator=bool(basis_generator),
                )
            else:
                logger.warning(
                    "Model checkpoint provided but no components could be loaded.",
                    checkpoint=str(args.model_checkpoint),
                )
            if args.basis_generator in {"diffusion", "hybrid_diffusion"} and not isinstance(
                basis_generator, DiffusionBasisGenerator
            ):
                logger.error(
                    "Diffusion basis generator requested via --basis-generator, but checkpoint does not contain diffusion weights.",
                    checkpoint=str(args.model_checkpoint),
                )
                return 1

        if args.basis_generator == "none":
            basis_generator = None
        if basis_generator is None and args.basis_generator == "mlp":
            choice = args.geo_encoder.lower() if hasattr(args, "geo_encoder") else "egnn"
            if choice == "simple":
                geo_encoder = SimpleGeoEncoder()
            else:
                try:
                    geo_encoder = GeoEncoder()
                except Exception:
                    geo_encoder = SimpleGeoEncoder()
            basis_generator = MLPBasisGenerator()
        if args.basis_generator in {"diffusion", "hybrid_diffusion"} and basis_generator is None:
            intensive_flag = bool(getattr(args, "intensive", False) or os.getenv("EDE_IMAGES_INTENSIVE", "0") == "1")
            cfg = DiffusionGeneratorConfig(
                k_max=64 if intensive_flag else 32,
                n_steps=64 if intensive_flag else 32,
                hidden_dim=256 if intensive_flag else 128,
                n_layers=6 if intensive_flag else 4,
                n_heads=8 if intensive_flag else 4,
            )
            basis_generator = DiffusionBasisGenerator(cfg)
            logger.warning(
                "Using a freshly-initialized diffusion generator (no checkpoint). Weights are random; treat outputs as exploratory."
            )

        aug_cfg = AugLagrangeConfig() if args.aug_boundary else None
        operator_mode = args.operator_mode
        gen_mode = args.basis_generator_mode
        if args.basis_generator in {"diffusion", "hybrid_diffusion"}:
            gen_mode = args.basis_generator

        system = discover_images(
            spec=spec,
            basis_types=basis_types,
            n_max=args.nmax,
            reg_l1=args.reg_l1,
            restarts=args.restarts,
            logger=logger,
            solver=solver_choice,
            solver_explicit=bool(solver_explicit),
            operator_mode=operator_mode,
            lista_model=lista_model,
            aug_lagrange=aug_cfg,
            adaptive_collocation_rounds=args.adaptive_collocation_rounds,
            n_points_override=args.n_points,
            ratio_boundary_override=args.ratio_boundary,
            lambda_group=lambda_group,
            basis_generator=basis_generator,
            basis_generator_mode=gen_mode,
            geo_encoder=geo_encoder,
            model_checkpoint=args.model_checkpoint,
            subtract_physical_potential=subtract_physical,
            intensive=intensive,
        )
        save_path = out_dir / "discovered_system.json"
        metadata = {
            "spec_path": str(Path(args.spec).resolve()),
            "basis_types": basis_types,
            "n_max": int(args.nmax),
            "reg_l1": float(args.reg_l1),
            "restarts": int(args.restarts),
            "basis_generator": args.basis_generator,
            "basis_generator_mode": args.basis_generator_mode,
            "subtract_physical": subtract_physical,
            "solver": solver_choice,
            "lambda_group": float(lambda_group),
        }
        save_image_system(system, save_path, metadata=metadata)
        manifest = metadata.copy()
        manifest_path = out_dir / "discovery_manifest.json"
        manifest["numeric_status"] = system.metadata.get("numeric_status", "ok")
        manifest["rel_resid"] = system.metadata.get("rel_resid")
        manifest["max_abs_weight"] = system.metadata.get("max_abs_weight")
        manifest["min_nonzero_weight"] = system.metadata.get("min_nonzero_weight")
        manifest.setdefault("condition_status", None)
        manifest.setdefault("gate1_status", None)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(
            "Images discovery completed.",
            out=str(save_path),
            n_images=len(system.elements),
        )
        return 0
    except NotImplementedError as exc:
        logger.error(
            "Images discovery not fully implemented for this configuration.",
            error=str(exc),
        )
        return 1
    except Exception as exc:
        logger.error(
            "Images discovery failed.",
            error=str(exc),
            exc_info=True,
        )
        return 1
    finally:
        logger.close()


def run_eval(args: argparse.Namespace) -> int:
    # Placeholder for a future evaluation path that would compare a
    # saved image system against high-fidelity BEM or analytic oracles.
    print(
        "[experimental] Image-system evaluation is not implemented yet.",
        file=sys.stderr,
    )
    return 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Electrodrive sparse Method-of-Images discovery CLI",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Discover subcommand
    p_disc = subparsers.add_parser(
        "discover",
        help="Run sparse image discovery for a given CanonicalSpec.",
    )
    p_disc.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to a CanonicalSpec JSON file.",
    )
    p_disc.add_argument(
        "--basis",
        type=str,
        default="point",
        help="Comma-separated list of basis types (e.g. 'point').",
    )
    p_disc.add_argument(
        "--nmax",
        type=int,
        default=16,
        help="Maximum number of images to retain after sparsification.",
    )
    p_disc.add_argument(
        "--reg-l1",
        type=float,
        default=1e-3,
        help="L1 regularisation strength for the ISTA sparse solver.",
    )
    p_disc.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Override number of collocation points (else EDE_IMAGES_N_POINTS or 512).",
    )
    p_disc.add_argument(
        "--ratio-boundary",
        type=float,
        default=None,
        help="Override boundary / interior fraction (else EDE_IMAGES_RATIO_BOUNDARY or 0.5).",
    )
    p_disc.add_argument(
        "--solver",
        type=str,
        choices=["ista", "lista"],
        default=None,
        help="Sparse solver backend.",
    )
    p_disc.add_argument(
        "--operator-mode",
        action="store_true",
        default=None,
        help="Use operator-form dictionary (BasisOperator) instead of dense matrix.",
    )
    p_disc.add_argument(
        "--adaptive-collocation-rounds",
        type=int,
        default=None,
        help="Number of residual-driven collocation rounds (1 = no adaptation).",
    )
    p_disc.add_argument(
        "--aug-boundary",
        action="store_true",
        help="Enable augmented Lagrangian boundary enforcement.",
    )
    p_disc.add_argument(
        "--subtract-physical",
        action="store_true",
        help="Subtract free-space potential from targets (induced solve).",
    )
    p_disc.add_argument(
        "--lambda-group",
        type=float,
        default=0.0,
        help="Group sparsity penalty (0 = no group lasso).",
    )
    p_disc.add_argument(
        "--restarts",
        type=int,
        default=None,
        help="If >0, enable LBFGS refinement of weights/positions.",
    )
    p_disc.add_argument(
        "--intensive",
        action="store_true",
        help="Use aggressive GPU-heavy settings (more points, candidates, and refinement).",
    )
    p_disc.add_argument(
        "--basis-generator",
        type=str,
        default="none",
        choices=["none", "mlp", "diffusion", "hybrid_diffusion"],
        help="Optional learned candidate generator. 'diffusion' and 'hybrid_diffusion' require a trained checkpoint.",
    )
    p_disc.add_argument(
        "--geo-encoder",
        type=str,
        default="egnn",
        choices=["egnn", "simple"],
        help="Geo encoder backbone to pair with the learned basis generator.",
    )
    p_disc.add_argument(
        "--basis-generator-mode",
        type=str,
        default="static_only",
        choices=["static_only", "static_plus_learned", "learned_only", "diffusion", "hybrid_diffusion"],
        help="How to combine learned candidates with static heuristics.",
    )
    p_disc.add_argument(
        "--out",
        type=str,
        default="runs/images_discover",
        help="Output directory for logs and discovered systems.",
    )
    p_disc.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Optional path to a learned model checkpoint (GeoEncoder, BasisGenerator, LISTA).",
    )
    p_disc.set_defaults(func=run_discover)

    # Eval subcommand (stub)
    p_eval = subparsers.add_parser(
        "eval",
        help="Evaluate a saved image system against a spec (experimental).",
    )
    p_eval.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to a CanonicalSpec JSON file.",
    )
    p_eval.add_argument(
        "--system",
        type=str,
        required=True,
        help="Path to a discovered_system.json file.",
    )
    p_eval.set_defaults(func=run_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
