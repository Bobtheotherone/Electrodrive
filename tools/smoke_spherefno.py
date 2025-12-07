from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from electrodrive.orchestration.spec_registry import stage0_sphere_external_path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from electrodrive.learn.spherefno_smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a SphereFNO surrogate against analytic Stage-0 sphere oracle.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SphereFNO checkpoint.")
    parser.add_argument("--spec", type=str, default=str(stage0_sphere_external_path()), help="Path to Stage-0 spec JSON.")
    parser.add_argument("--device", type=str, default="", help="Torch device (default: auto).")
    parser.add_argument("--samples", type=int, default=8, help="Number of evaluation samples.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_smoke(Path(args.spec), Path(args.ckpt), device=device, n_samples=int(args.samples))


if __name__ == "__main__":
    raise SystemExit(main())
