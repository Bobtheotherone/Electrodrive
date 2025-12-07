#!/usr/bin/env python3
import sys
import traceback
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

from electrodrive.utils.logging import JsonlLogger


def main() -> None:
    # Always resolve relative to the repo that this file lives in
    repo_root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "runs" / f"logger_smoketest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lg = JsonlLogger(out_dir)

    # 1) Simple structured message
    lg.info(
        "Hello world from logger smoketest.",
        message_type="simple",
        answer=42,
    )

    # 2) Simulated exception with manual traceback payload
    try:
        1 / 0
    except Exception as e:
        tb = traceback.format_exc()
        lg.error(
            "Captured exception in logger smoketest.",
            error=str(e),
            traceback=tb,
        )

    # 3) Structured + numpy (if available)
    arr = np.arange(5).tolist() if np is not None else [0, 1, 2, 3, 4]
    lg.warning(
        "Array test field",
        array_values=arr,
        message_type="numpy_test",
    )

    lg.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
