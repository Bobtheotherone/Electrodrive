from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from electrodrive.learn.spherefno_train import main as _main


def cli() -> int:
    parser = argparse.ArgumentParser(description="Train SphereFNO surrogate on Stage-0 sphere axis data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "train_spherefno_stage0.yaml"),
        help="YAML config with dataset/model/training parameters.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("runs") / "spherefno_stage0"),
        help="Output directory for checkpoints and logs.",
    )
    args = parser.parse_args()
    best_path = _main(Path(args.config), Path(args.out))
    return 0 if best_path.exists() else 1


def main() -> int:
    return cli()


if __name__ == "__main__":
    raise SystemExit(main())
