# electrodrive/learn/cli.py
from __future__ import annotations

from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path


def register_learn_commands(subparsers: _SubParsersAction) -> None:
    """Register 'train' and 'eval' subcommands on the main Electrodrive CLI.

    This is imported lazily from electrodrive.cli, so failures here do not affect
    the base solver CLI when learning dependencies are absent.
    """

    # Train command
    p_train: ArgumentParser = subparsers.add_parser(
        "train",
        help="Train a conditional electrostatics model (learning stack).",
    )
    p_train.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ExperimentConfig YAML.",
    )
    p_train.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs.",
    )
    p_train.set_defaults(func=_main_train)

    # Eval command
    p_eval: ArgumentParser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained model using certify-style gates.",
    )
    p_eval.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ExperimentConfig YAML (for eval settings).",
    )
    p_eval.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint to evaluate.",
    )
    p_eval.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for evaluation reports.",
    )
    p_eval.set_defaults(func=_main_eval)

    # MOI 3.0 bilevel training (unrolled LISTA)
    try:
        from electrodrive.learn import mo3_train as _mo3_train

        p_mo3: ArgumentParser = subparsers.add_parser(
            "mo3_train",
            help="Stage-0/1 MOI-3.0 bilevel training with learned solver.",
        )
        _mo3_train.add_arguments(p_mo3)
        p_mo3.set_defaults(func=_main_mo3_train)
    except Exception:
        # Keep the CLI resilient if optional deps are missing.
        pass

    # SphereFNO Stage-0 training
    try:
        p_sphere: ArgumentParser = subparsers.add_parser(
            "train_spherefno_stage0",
            help="Train SphereFNO surrogate for Stage-0 on-axis grounded sphere.",
        )
        p_sphere.add_argument(
            "--config",
            type=str,
            default=str(Path("configs") / "train_spherefno_stage0.yaml"),
            help="YAML config for SphereFNO training.",
        )
        p_sphere.add_argument(
            "--out",
            type=str,
            default=str(Path("runs") / "spherefno_stage0"),
            help="Output directory for checkpoints/logs.",
        )
        p_sphere.set_defaults(func=_main_train_spherefno)
    except Exception:
        pass

    # SphereFNO smoke test
    try:
        p_smoke: ArgumentParser = subparsers.add_parser(
            "smoke_spherefno",
            help="Smoke test a SphereFNO checkpoint against analytic Stage-0 oracle.",
        )
        p_smoke.add_argument(
            "--ckpt",
            type=str,
            required=True,
            help="Path to SphereFNO checkpoint.",
        )
        p_smoke.add_argument(
            "--spec",
            type=str,
            default=str(Path("specs") / "sphere_axis_point_external.json"),
            help="Stage-0 spec JSON path.",
        )
        p_smoke.add_argument(
            "--device",
            type=str,
            default="",
            help="Torch device override.",
        )
        p_smoke.add_argument(
            "--samples",
            type=int,
            default=8,
            help="Number of evaluation samples.",
        )
        p_smoke.set_defaults(func=_main_smoke_spherefno)
    except Exception:
        pass


def _main_train(args) -> int:
    # Heavy imports moved inside to keep register_learn_commands light
    from electrodrive.learn.specs import ExperimentConfig
    from electrodrive.learn.train import train as _train_entry

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = Path(args.out)
    return _train_entry(cfg, out_dir)


def _main_eval(args) -> int:
    # Heavy imports moved inside to keep register_learn_commands light
    from electrodrive.learn.specs import ExperimentConfig
    from electrodrive.learn.eval import (
        run_evaluation as _eval_entry,
    )

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = Path(args.out)
    ckpt_path = Path(args.ckpt)
    return _eval_entry(cfg, ckpt_path, out_dir)


def _main_mo3_train(args) -> int:
    # Import inside to avoid optional dependency issues on minimal installs.
    from electrodrive.learn import mo3_train

    return mo3_train.run_from_namespace(args)


def _main_train_spherefno(args) -> int:
    from electrodrive.learn import spherefno_train

    spherefno_train.main(Path(args.config), Path(args.out))
    return 0


def _main_smoke_spherefno(args) -> int:
    import torch

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from electrodrive.learn import spherefno_smoke

    return spherefno_smoke.run_smoke(Path(args.spec), Path(args.ckpt), device=device, n_samples=int(args.samples))
