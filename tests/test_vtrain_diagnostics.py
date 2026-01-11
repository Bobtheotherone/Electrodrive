import json
import math
from pathlib import Path

import torch

from electrodrive.experiments.vtrain_diagnostics import (
    build_vtrain_explosion_snapshot,
    write_vtrain_explosion_snapshot,
)
from electrodrive.orchestration.parser import CanonicalSpec


def _simple_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
            "conductors": [],
            "dielectrics": [],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.0]}],
            "BCs": "Dirichlet",
            "symmetry": [],
            "queries": [],
        }
    )


def test_vtrain_snapshot_payload(tmp_path: Path) -> None:
    spec = _simple_spec()
    X_train = torch.tensor([[0.0, 0.0, 1e-4], [0.2, 0.0, 0.2]], dtype=torch.float32)
    V_train = torch.tensor([1.0e12, -3.0], dtype=torch.float32)
    A_train = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    payload = build_vtrain_explosion_snapshot(
        spec,
        X_train,
        V_train,
        A_train,
        layered_reference_enabled=False,
        reference_subtracted_for_fit=False,
        nan_to_num_applied=False,
        clamp_applied=False,
        seed=123,
        gen=1,
        program_idx=7,
    )

    stats = payload["stats"]
    assert math.isclose(stats["min_distance_to_any_point_charge"], 1e-4, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(stats["max_abs_V_train"], 1.0e12, rel_tol=0.0, abs_tol=1e6)
    assert math.isclose(stats["max_abs_A_train"], 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert stats["frac_nonfinite_V_train"] == 0.0
    assert payload["flags"]["layered_reference_enabled"] is False
    assert payload["flags"]["reference_subtracted_for_fit"] is False

    topk = payload["topk_abs_V_train"]
    assert len(topk) == 2
    assert topk[0]["index"] == 0
    assert math.isclose(topk[0]["abs_value"], 1.0e12, rel_tol=0.0, abs_tol=1e6)
    assert all(
        math.isclose(val, exp, rel_tol=0.0, abs_tol=1e-6)
        for val, exp in zip(topk[0]["x"], [0.0, 0.0, 1e-4])
    )

    assert write_vtrain_explosion_snapshot(tmp_path, payload)
    assert not write_vtrain_explosion_snapshot(tmp_path, payload)
    loaded = json.loads((tmp_path / "preflight_vtrain_snapshot.json").read_text(encoding="utf-8"))
    assert loaded["stats"].keys() == stats.keys()
