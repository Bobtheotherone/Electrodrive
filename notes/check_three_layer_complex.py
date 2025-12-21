"""Lightweight GPU checker for three_layer_complex basis candidates."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch

from electrodrive.images.basis import generate_candidate_basis
from electrodrive.orchestration.parser import parse_spec


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required for three_layer_complex checker."
    device = torch.device("cuda")
    dtype = torch.float32

    spec_path = Path("specs/planar_three_layer_eps2_80_sym_h04_region1.json")
    spec = parse_spec(spec_path)
    basis_types = ["axis_point", "three_layer_images", "three_layer_complex"]

    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=16,
        device=device,
        dtype=dtype,
    )

    layered = [c for c in candidates if c.type == "three_layer_images"]
    families = [getattr(c, "_group_info", {}).get("family_name", "unknown") for c in layered]
    conductor_ids = [getattr(c, "_group_info", {}).get("conductor_id", None) for c in layered]

    summary = {
        "total": len(candidates),
        "layered_total": len(layered),
        "families": dict(Counter(families)),
        "conductor_ids": dict(Counter(conductor_ids)),
    }

    base = Path("runs/checks/three_layer_complex")
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run_{int(time.time())}"
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"run_{int(time.time())}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("three_layer_complex_checker_ok", json.dumps(summary))


if __name__ == "__main__":
    main()
