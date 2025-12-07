from __future__ import annotations

import os

import torch

from electrodrive.images.basis import (
    BASIS_FAMILY_ENUM,
    generate_candidate_basis,
    compute_group_ids,
)
from electrodrive.orchestration.parser import CanonicalSpec


def test_group_ids_are_stable_and_separated() -> None:
    """Group IDs stay stable across runs and separate conductor/family motifs."""
    os.environ["EDE_IMAGES_SHUFFLE_CANDIDATES"] = "0"
    spec = CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0},
                {"type": "sphere", "center": [3.0, 0.0, 0.0], "radius": 1.2},
                {"type": "plane", "z": -1.0, "potential": 0.0},
            ],
            "charges": [
                {"type": "point", "q": 1.0, "pos": [0.0, 0.0, 2.5]},
                {"type": "point", "q": -0.5, "pos": [3.0, 0.0, 3.5]},
            ],
        }
    )

    basis_types = ["sphere_kelvin_ladder", "sphere_equatorial_ring", "axis_point"]
    n_candidates = 32

    def _groups_for_run() -> torch.Tensor:
        cands = generate_candidate_basis(
            spec,
            basis_types=basis_types,
            n_candidates=n_candidates,
            device="cpu",
            dtype=torch.float32,
        )
        return compute_group_ids(cands, device=torch.device("cpu"), dtype=torch.long), cands

    groups1, cands1 = _groups_for_run()
    groups2, _ = _groups_for_run()

    assert torch.equal(groups1, groups2)

    families_seen = set()
    for gid, c in zip(groups1, cands1):
        info = getattr(c, "_group_info", {})
        cid = int(info.get("conductor_id", 0))
        fam = info.get("family", 0)
        motif = int(info.get("motif_index", 0))
        expected = cid * 1000 + fam * 10 + motif
        assert int(gid.item()) == expected
        families_seen.add(info.get("family_name", c.type))

    assert len(torch.unique(groups1)) >= 1
    assert len(families_seen) >= 2
