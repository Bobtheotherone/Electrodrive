from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from electrodrive.images.training import BilevelTrainConfig, train_stage0


def _split_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Training stage (0 or 1). Stage-2 is reserved for gratings.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of optimization steps (unrolled batches).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tasks per optimization step.",
    )
    parser.add_argument(
        "--basis-plane",
        type=str,
        default="mirror_stack,point",
        help="Comma-separated basis families for plane tasks.",
    )
    parser.add_argument(
        "--basis-sphere",
        type=str,
        default="sphere_kelvin_ladder,axis_point,point",
        help="Comma-separated basis families for sphere tasks.",
    )
    parser.add_argument(
        "--basis-dimer",
        type=str,
        default="sphere_kelvin_ladder,axis_point,point",
        help="Comma-separated basis families for Stage-1 sphere dimer tasks.",
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=64,
        help="Number of static (heuristic) candidates per task.",
    )
    parser.add_argument(
        "--n-learned",
        type=int,
        default=0,
        help="Number of learned candidates from BasisGenerator.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=512,
        help="Collocation points for the training head.",
    )
    parser.add_argument(
        "--n-points-val",
        type=int,
        default=512,
        help="Collocation points for validation loss.",
    )
    parser.add_argument(
        "--ratio-boundary",
        type=float,
        default=0.5,
        help="Boundary ratio for training collocation.",
    )
    parser.add_argument(
        "--ratio-boundary-val",
        type=float,
        default=0.5,
        help="Boundary ratio for validation collocation.",
    )
    parser.add_argument(
        "--lambda-bc",
        type=float,
        default=50.0,
        help="Boundary loss weight.",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=1e-4,
        help="Outer L1 penalty on raw LISTA weights.",
    )
    parser.add_argument(
        "--lambda-group",
        type=float,
        default=0.0,
        help="Group LASSO penalty strength inside LISTA.",
    )
    parser.add_argument(
        "--lista-steps",
        type=int,
        default=10,
        help="Unrolled LISTA iterations.",
    )
    parser.add_argument(
        "--lista-rank",
        type=int,
        default=0,
        help="Optional low-rank correction rank for diagonal LISTA.",
    )
    parser.add_argument(
        "--lr-lista",
        type=float,
        default=1e-3,
        help="Learning rate for LISTA parameters.",
    )
    parser.add_argument(
        "--lr-geo",
        type=float,
        default=3e-4,
        help="Learning rate for geometry encoder / basis generator.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global gradient norm clip.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (cuda, cuda:0, cpu). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 | float64 | bfloat16 | float16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Global RNG seed.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Checkpoint frequency in steps (0 disables mid-run checkpoints).",
    )
    parser.add_argument(
        "--no-stage1-variants",
        action="store_true",
        help="Disable Stage-1 spec variants; train only on the canonical lens.",
    )


def _cfg_from_args(args: argparse.Namespace) -> BilevelTrainConfig:
    return BilevelTrainConfig(
        out_dir=Path(args.out),
        stage=int(args.stage),
        max_steps=int(args.steps),
        batch_size=int(args.batch_size),
        n_candidates_static=int(args.n_static),
        n_candidates_learned=int(args.n_learned),
        basis_plane=_split_list(args.basis_plane),
        basis_sphere=_split_list(args.basis_sphere),
        basis_dimer=_split_list(args.basis_dimer),
        n_points_train=int(args.n_points),
        n_points_val=int(args.n_points_val),
        ratio_boundary_train=float(args.ratio_boundary),
        ratio_boundary_val=float(args.ratio_boundary_val),
        lambda_bc=float(args.lambda_bc),
        lambda_l1=float(args.lambda_l1),
        lambda_group=float(args.lambda_group),
        lista_steps=int(args.lista_steps),
        lista_rank=int(args.lista_rank),
        lr_lista=float(args.lr_lista),
        lr_geo=float(args.lr_geo),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        device=args.device,
        dtype=str(args.dtype),
        seed=int(args.seed),
        checkpoint_every=int(args.checkpoint_every),
        stage1_include_variants=not bool(getattr(args, "no_stage1_variants", False)),
    )


def run_from_namespace(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    train_stage0(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MOI Discovery 3.0 Stage-0 bilevel training (unrolled LISTA)."
    )
    add_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    raise SystemExit(main())
