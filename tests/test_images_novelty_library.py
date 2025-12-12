import json
from pathlib import Path

import torch

from electrodrive.discovery import novelty
from electrodrive.images.basis import PointChargeBasis, annotate_group_info
from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec


def _spec():
    return CanonicalSpec.from_json(
        {
            "domain": {"bbox": [[-1, -1, -1], [1, 1, 1]]},
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.5, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.5},
            ],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
            "BCs": "dielectric_interfaces",
        }
    )


def test_library_loader_uses_real_entries(monkeypatch):
    spec = _spec()
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, -0.1])}, type_name="three_layer_images")
    annotate_group_info(elem, conductor_id=1, family_name="three_layer_slab", motif_index=0)
    system = ImageSystem([elem], torch.tensor([1.0]))

    def fake_loader():
        return [(spec, system)]

    monkeypatch.setattr(novelty, "_load_library_configs_from_docs", fake_loader)
    novelty._LIBRARY_CACHE.clear()
    novelty._FLATTEN_CACHE.clear()
    fps = novelty._default_library_fingerprints()
    assert len(fps) == 1
    score = novelty.novelty_score(fps[0])
    assert 0.0 <= score <= 1.0


def test_library_loader_fallback(monkeypatch):
    def empty_loader():
        return []

    monkeypatch.setattr(novelty, "_load_library_configs_from_docs", empty_loader)
    novelty._LIBRARY_CACHE.clear()
    novelty._FLATTEN_CACHE.clear()
    fps = novelty._default_library_fingerprints()
    assert len(fps) >= 1
