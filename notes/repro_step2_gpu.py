"""GPU-first repro for current three-layer basis heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch

from electrodrive.images.basis import ImageBasisElement, generate_candidate_basis
from electrodrive.orchestration.parser import parse_spec


def _positions(elems: Iterable[ImageBasisElement]) -> list[Tuple[float, float, float]]:
    pts = []
    for elem in elems:
        pos = elem.params.get("position")
        if pos is None:
            continue
        pts.append(tuple(pos.detach().cpu().tolist()))
    return pts


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required for repro (GPU-first rule)."
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

    axis = [c for c in candidates if c.type == "axis_point"]
    layered = [c for c in candidates if c.type == "three_layer_images"]

    print("spec:", spec_path.name)
    print("device:", device)
    print("dtype:", dtype)
    print("total candidates:", len(candidates))
    print("axis_point count:", len(axis))
    print("three_layer_images count:", len(layered))
    print("\naxis_point z values (sorted):")
    print(sorted(p[2] for p in _positions(axis)))
    print("\nthree_layer_images positions (x, y, z):")
    for elem in layered:
        pos = elem.params["position"].detach().cpu().tolist()
        group = getattr(elem, "_group_info", {}) or {}
        family = group.get("family_name", "unknown")
        motif = group.get("motif_index", None)
        cid = group.get("conductor_id", None)
        print((pos[0], pos[1], pos[2]), f"family={family}", f"motif={motif}", f"cid={cid}")


if __name__ == "__main__":
    main()
