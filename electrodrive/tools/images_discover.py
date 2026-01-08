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
from electrodrive.images.search import (
    AugLagrangeConfig,
    discover_images,
    assemble_basis_matrix,
    get_collocation_data,
    solve_sparse,
)
from electrodrive.images.optim import (
    ADMMConfig,
    ConstraintSpec,
    OuterSolveConfig,
    SparseSolveRequest,
    optimize_theta_adam,
    optimize_theta_lbfgs,
    refine_and_certify,
    search as global_search,
)
from electrodrive.images.basis import compute_group_ids
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


def _parse_mode_indices(text: Optional[str]) -> List[tuple[int, int]]:
    if not text:
        return []
    tokens = text.replace(";", " ").split()
    modes: List[tuple[int, int]] = []
    for tok in tokens:
        parts = [p for p in tok.split(",") if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid mode token: {tok}")
        modes.append((int(parts[0]), int(parts[1])))
    return modes


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

    outer_choice = str(getattr(args, "outer_opt", "none") or "none").strip().lower()
    global_choice = str(getattr(args, "global_search", "none") or "none").strip().lower()
    certify_fp64 = bool(getattr(args, "certify_fp64", False))
    certify_mpmath = bool(getattr(args, "certify_mpmath", False))
    budget = int(getattr(args, "budget", 0) or 0)
    seed = getattr(args, "seed", None)

    constraint_specs = None
    constraint_mode = "none"
    if getattr(args, "constraint", None):
        constraint_specs = []
        mode_indices = _parse_mode_indices(getattr(args, "fft_modes", None))
        for idx, basis in enumerate(args.constraint):
            params = {}
            region = None
            if basis == "collocation":
                region = getattr(args, "constraint_region", None)
                if region == "all":
                    region = None
            elif basis == "planar_fft":
                if args.fft_grid_h is None or args.fft_grid_w is None:
                    raise ValueError("--fft-grid-h/--fft-grid-w are required for planar_fft constraints.")
                params["grid_shape"] = (int(args.fft_grid_h), int(args.fft_grid_w))
                if mode_indices:
                    params["mode_indices"] = mode_indices
                params["fft_shift"] = bool(getattr(args, "fft_shift", False))
            elif basis == "sphere_sh":
                if args.sh_lmax is None:
                    raise ValueError("--sh-lmax is required for sphere_sh constraints.")
                params["Lmax"] = int(args.sh_lmax)
            else:
                raise ValueError(f"Unsupported constraint basis: {basis}")
            constraint_specs.append(
                ConstraintSpec(
                    name=f"{basis}_{idx}",
                    kind=str(getattr(args, "constraint_kind", "eq")),
                    weight=float(getattr(args, "constraint_weight", 1.0)),
                    eps=float(getattr(args, "constraint_eps", 0.0)),
                    region=region,
                    basis=basis,
                    params=params,
                )
            )
        constraint_mode = str(getattr(args, "constraint_mode", "admm")).strip().lower() or "admm"

    admm_cfg = None
    if solver_choice == "admm_constrained":
        admm_kwargs = {}
        if args.admm_rho is not None:
            admm_kwargs["rho"] = float(args.admm_rho)
        if args.admm_max_iter is not None:
            admm_kwargs["max_iter"] = int(args.admm_max_iter)
        if args.admm_tol is not None:
            admm_kwargs["tol"] = float(args.admm_tol)
        if args.admm_w_update_iters is not None:
            admm_kwargs["w_update_iters"] = int(args.admm_w_update_iters)
        if args.admm_unroll_steps is not None:
            admm_kwargs["unroll_steps"] = int(args.admm_unroll_steps)
        if admm_kwargs:
            admm_cfg = ADMMConfig(**admm_kwargs)

    gfn_checkpoint = getattr(args, "gfn_checkpoint", None)
    gfn_seed = getattr(args, "gfn_seed", None)
    flow_checkpoint = getattr(args, "flow_checkpoint", None)
    flow_steps = getattr(args, "flow_steps", None)
    flow_solver = getattr(args, "flow_solver", None)
    flow_temp = getattr(args, "flow_temp", None)
    flow_dtype = getattr(args, "flow_dtype", None)
    flow_seed = getattr(args, "flow_seed", None)
    allow_random_flow = bool(getattr(args, "allow_random_flow", False))
    logger.info(
        "Images discovery run started.",
        spec=str(args.spec),
        basis=args.basis,
        nmax=int(args.nmax),
        reg_l1=float(args.reg_l1),
        restarts=int(args.restarts),
        basis_generator=args.basis_generator,
        basis_generator_mode=args.basis_generator_mode,
        gfn_checkpoint=gfn_checkpoint,
        gfn_seed=gfn_seed,
        flow_checkpoint=flow_checkpoint,
        flow_steps=flow_steps,
        flow_solver=flow_solver,
        flow_temp=flow_temp,
        flow_dtype=flow_dtype,
        flow_seed=flow_seed,
        allow_random_flow=bool(allow_random_flow),
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
        gfn_mode = args.basis_generator in {"gfn", "gfn_flow"} or args.basis_generator_mode in {"gfn", "gfn_flow"}
        gfn_flow_mode = args.basis_generator == "gfn_flow" or args.basis_generator_mode == "gfn_flow"
        if gfn_flow_mode:
            if not gfn_checkpoint:
                raise SystemExit("gfn_flow requires --gfn-checkpoint; random weights are not allowed.")
            if not flow_checkpoint and not allow_random_flow:
                raise SystemExit(
                    "gfn_flow requires --flow-checkpoint unless --allow-random-flow is set."
                )
        elif gfn_mode and not gfn_checkpoint:
            logger.error(
                "GFlowNet generator requires a checkpoint; random weights are not allowed.",
            )
            return 1

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
        if gfn_mode:
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
        if args.basis_generator in {"gfn", "gfn_flow"}:
            gen_mode = args.basis_generator

        gfdsl_program_dir = getattr(args, "gfdsl_program_dir", None)
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
            gfdsl_program_dir=gfdsl_program_dir,
            geo_encoder=geo_encoder,
            model_checkpoint=args.model_checkpoint,
            gfn_checkpoint=gfn_checkpoint,
            gfn_seed=gfn_seed,
            flow_checkpoint=flow_checkpoint,
            flow_steps=flow_steps,
            flow_solver=flow_solver,
            flow_temp=flow_temp,
            flow_dtype=flow_dtype,
            flow_seed=flow_seed,
            allow_random_flow=allow_random_flow,
            subtract_physical_potential=subtract_physical,
            intensive=intensive,
            constraint_specs=constraint_specs,
            admm_cfg=admm_cfg,
            constraint_mode=constraint_mode,
        )

        needs_outer = outer_choice not in {"", "none"}
        needs_global = global_choice not in {"", "none"}
        if (needs_outer or needs_global or certify_fp64) and system.elements:
            logger.info(
                "Outer/global optimization requested; using experimental column-scale parameterization.",
                outer_opt=outer_choice,
                global_search=global_choice,
                certify_fp64=bool(certify_fp64),
            )
            need_boundary = bool(constraint_specs or certify_fp64)
            colloc_out = get_collocation_data(
                spec,
                logger,
                device=system.device,
                dtype=system.dtype,
                return_is_boundary=need_boundary,
                n_points_override=args.n_points,
                ratio_override=args.ratio_boundary,
                subtract_physical_potential=subtract_physical,
            )
            if isinstance(colloc_out, tuple) and len(colloc_out) == 3:
                colloc_pts, target_vec, is_boundary = colloc_out  # type: ignore[misc]
            else:
                colloc_pts, target_vec = colloc_out  # type: ignore[misc]
                is_boundary = None

            if colloc_pts.numel() > 0 and target_vec.numel() > 0:
                A_base = assemble_basis_matrix(system.elements, colloc_pts)

                class _ColumnScaleObjective:
                    def __init__(self, A: torch.Tensor, X: torch.Tensor, g: torch.Tensor) -> None:
                        self.A = A
                        self.X = X
                        self.g = g

                    def build_dictionary(self, theta: torch.Tensor):
                        scale = 1.0 + theta.view(1, 1)
                        A_scaled = self.A * scale
                        return A_scaled, self.X, self.g, {"A": A_scaled}

                    def loss(self, theta: torch.Tensor, w: torch.Tensor, metadata: dict):
                        A = metadata.get("A", self.A)
                        pred = A.matmul(w)
                        return torch.mean((pred - self.g) ** 2)

                    def constraints(self, theta: torch.Tensor):
                        return None

                objective = _ColumnScaleObjective(A_base, colloc_pts, target_vec)
                theta_init = torch.zeros(1, device=system.device, dtype=system.dtype)
                inner_solver = "admm_constrained" if constraint_specs else "implicit_lasso"
                inner_admm_cfg = admm_cfg
                if inner_solver == "admm_constrained" and needs_outer:
                    base_cfg = admm_cfg if isinstance(admm_cfg, ADMMConfig) else ADMMConfig()
                    inner_admm_cfg = ADMMConfig(
                        rho=base_cfg.rho,
                        rho_growth=base_cfg.rho_growth,
                        max_rho=base_cfg.max_rho,
                        max_iter=base_cfg.max_iter,
                        tol=base_cfg.tol,
                        unroll_steps=max(5, int(base_cfg.unroll_steps) if base_cfg.unroll_steps > 0 else 5),
                        w_update_iters=base_cfg.w_update_iters,
                        diff_mode=base_cfg.diff_mode,
                        verbose=base_cfg.verbose,
                        track_residuals=base_cfg.track_residuals,
                    )
                solve_cfg = OuterSolveConfig(
                    solver=inner_solver,
                    reg_l1=float(args.reg_l1),
                    max_iter=200,
                    tol=1e-6,
                    lambda_group=float(lambda_group),
                    group_ids=compute_group_ids(system.elements, device=system.device, dtype=torch.long),
                    normalize_columns=True,
                    constraints=constraint_specs,
                    constraint_mode=constraint_mode,
                    admm_cfg=inner_admm_cfg,
                )

                theta_best = theta_init
                if needs_global and budget > 0:
                    theta_best, report = global_search(
                        theta_best,
                        objective,
                        solve_cfg,
                        method=global_choice,
                        budget=budget,
                        seed=seed,
                    )
                    logger.info(
                        "Global search complete.",
                        method=report.method,
                        best_loss=float(report.best_loss),
                        evaluations=int(report.evaluations),
                    )

                if needs_outer:
                    if outer_choice == "lbfgs":
                        result = optimize_theta_lbfgs(
                            theta_best,
                            objective,
                            solve_cfg,
                            max_iter=25,
                            seed=seed,
                            restarts=1,
                        )
                    else:
                        result = optimize_theta_adam(
                            theta_best,
                            objective,
                            solve_cfg,
                            steps=40,
                            lr=5e-2,
                            seed=seed,
                            restarts=1,
                        )
                    theta_best = result.theta
                    logger.info(
                        "Outer optimization complete.",
                        method=outer_choice,
                        loss=float(result.loss),
                    )

                A_opt, _, g_opt, _ = objective.build_dictionary(theta_best)
                w_opt, _, stats = solve_sparse(
                    A_opt,
                    colloc_pts,
                    g_opt,
                    is_boundary if need_boundary else None,
                    logger,
                    reg_l1=float(args.reg_l1),
                    solver=inner_solver,
                    group_ids=solve_cfg.group_ids,
                    lambda_group=float(lambda_group),
                    normalize_columns=True,
                    constraints=constraint_specs,
                    constraint_mode=constraint_mode,
                    admm_cfg=inner_admm_cfg,
                    return_stats=True,
                )
                system.weights = w_opt
                system.metadata["outer_theta"] = float(theta_best.detach().view(-1)[0].item())
                system.metadata["outer_stats"] = stats

                if certify_fp64:
                    req = SparseSolveRequest(
                        A=A_opt,
                        X=colloc_pts,
                        g=g_opt,
                        is_boundary=is_boundary if need_boundary else None,
                        lambda_l1=float(args.reg_l1),
                        lambda_group=float(lambda_group),
                        group_ids=solve_cfg.group_ids,
                        weight_prior=None,
                        lambda_weight_prior=0.0,
                        normalize_columns=True,
                        col_norms=None,
                        constraints=constraint_specs or [],
                        max_iter=200,
                        tol=1e-6,
                        warm_start=w_opt,
                        return_stats=True,
                        dtype_policy=None,
                    )
                    w64, cert = refine_and_certify(
                        req,
                        solver=inner_solver,
                        admm_cfg=inner_admm_cfg,
                        w_init=w_opt,
                        use_mpmath=certify_mpmath,
                    )
                    system.weights = w64
                    system.metadata["certificate"] = cert
                    logger.info("FP64 certification complete.", certificate=cert)
        save_path = out_dir / "discovered_system.json"
        metadata = {
            "spec_path": str(Path(args.spec).resolve()),
            "basis_types": basis_types,
            "n_max": int(args.nmax),
            "reg_l1": float(args.reg_l1),
            "restarts": int(args.restarts),
            "basis_generator": args.basis_generator,
            "basis_generator_mode": args.basis_generator_mode,
            "gfn_checkpoint": gfn_checkpoint,
            "flow_checkpoint": flow_checkpoint,
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
        help="Comma-separated list of basis types (e.g. 'point' or 'point,three_layer_complex').",
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
        choices=["ista", "lista", "implicit_lasso", "implicit_grouplasso", "admm_constrained"],
        default=None,
        help="Sparse solver backend.",
    )
    p_disc.add_argument(
        "--constraint",
        type=str,
        action="append",
        choices=["collocation", "planar_fft", "sphere_sh"],
        default=None,
        help="Add a constraint block (repeatable).",
    )
    p_disc.add_argument(
        "--constraint-kind",
        type=str,
        choices=["eq", "l2", "linf"],
        default="eq",
        help="Constraint type (equality, l2 bound, or linf bound).",
    )
    p_disc.add_argument(
        "--constraint-eps",
        type=float,
        default=0.0,
        help="Constraint bound (eps).",
    )
    p_disc.add_argument(
        "--constraint-weight",
        type=float,
        default=1.0,
        help="Constraint weight multiplier.",
    )
    p_disc.add_argument(
        "--constraint-region",
        type=str,
        choices=["all", "boundary", "interior"],
        default="all",
        help="Collocation region selection for collocation constraints.",
    )
    p_disc.add_argument(
        "--constraint-mode",
        type=str,
        choices=["none", "admm"],
        default="admm",
        help="Constraint handling mode.",
    )
    p_disc.add_argument(
        "--fft-grid-h",
        type=int,
        default=None,
        help="Planar FFT grid height.",
    )
    p_disc.add_argument(
        "--fft-grid-w",
        type=int,
        default=None,
        help="Planar FFT grid width.",
    )
    p_disc.add_argument(
        "--fft-modes",
        type=str,
        default=None,
        help="Planar FFT mode indices as 'ky,kx;ky,kx'.",
    )
    p_disc.add_argument(
        "--fft-shift",
        action="store_true",
        help="Apply fftshift before selecting planar FFT modes.",
    )
    p_disc.add_argument(
        "--sh-lmax",
        type=int,
        default=None,
        help="Spherical harmonics maximum degree (Lmax).",
    )
    p_disc.add_argument(
        "--admm-rho",
        type=float,
        default=None,
        help="ADMM rho penalty parameter.",
    )
    p_disc.add_argument(
        "--admm-max-iter",
        type=int,
        default=None,
        help="ADMM maximum iterations.",
    )
    p_disc.add_argument(
        "--admm-tol",
        type=float,
        default=None,
        help="ADMM convergence tolerance.",
    )
    p_disc.add_argument(
        "--admm-w-update-iters",
        type=int,
        default=None,
        help="Inner w-update iterations per ADMM step.",
    )
    p_disc.add_argument(
        "--admm-unroll-steps",
        type=int,
        default=None,
        help="Unrolled ADMM steps for training mode.",
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
        "--outer-opt",
        "--outer_opt",
        dest="outer_opt",
        choices=["lbfgs", "adam", "none"],
        default="none",
        help="Outer optimization method for experimental continuous parameters.",
    )
    p_disc.add_argument(
        "--global-search",
        "--global_search",
        dest="global_search",
        choices=["cmaes", "basinhop", "multistart", "none"],
        default="none",
        help="Global search strategy for experimental continuous parameters.",
    )
    p_disc.add_argument(
        "--budget",
        type=int,
        default=16,
        help="Evaluation budget for global search.",
    )
    p_disc.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for outer/global optimization.",
    )
    p_disc.add_argument(
        "--certify-fp64",
        "--certify_fp64",
        dest="certify_fp64",
        action="store_true",
        help="Re-solve weights in float64 and emit certification stats.",
    )
    p_disc.add_argument(
        "--certify-mpmath",
        "--certify_mpmath",
        dest="certify_mpmath",
        action="store_true",
        help="Enable optional mpmath verification for tiny problems.",
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
        choices=["none", "mlp", "diffusion", "hybrid_diffusion", "gfn", "gfn_flow"],
        help="Optional learned candidate generator. 'diffusion', 'gfn', and 'gfn_flow' require checkpoints.",
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
        choices=["static_only", "static_plus_learned", "learned_only", "diffusion", "hybrid_diffusion", "gfn", "gfn_flow", "gfdsl"],
        help="How to combine learned candidates with static heuristics.",
    )
    p_disc.add_argument(
        "--gfdsl-program-dir",
        type=str,
        default=None,
        help="Directory containing GFDSL program JSON files (basis_generator_mode=gfdsl).",
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
    p_disc.add_argument(
        "--gfn-checkpoint",
        type=str,
        default=None,
        help="Path to a GFlowNet generator checkpoint (required for gfn mode).",
    )
    p_disc.add_argument(
        "--flow-checkpoint",
        type=str,
        default=None,
        help="Path to a flow sampler checkpoint (required for gfn_flow unless --allow-random-flow).",
    )
    p_disc.add_argument(
        "--flow-steps",
        type=int,
        default=None,
        help="Number of flow integration steps (1-8 typical).",
    )
    p_disc.add_argument(
        "--flow-solver",
        type=str,
        choices=["euler", "heun", "rk4"],
        default=None,
        help="Flow ODE solver.",
    )
    p_disc.add_argument(
        "--flow-temp",
        type=float,
        default=None,
        help="Flow sampling temperature.",
    )
    p_disc.add_argument(
        "--flow-dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Flow sampling dtype.",
    )
    p_disc.add_argument(
        "--flow-seed",
        type=int,
        default=None,
        help="Optional RNG seed for flow parameter sampling.",
    )
    p_disc.add_argument(
        "--allow-random-flow",
        action="store_true",
        help="Allow gfn_flow to run with random flow weights if no checkpoint is provided.",
    )
    p_disc.add_argument(
        "--gfn-seed",
        type=int,
        default=None,
        help="Optional RNG seed for GFlowNet program sampling.",
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
