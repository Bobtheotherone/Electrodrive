import json
from pathlib import Path

from electrodrive.images.search import _extract_slab_bounds
from electrodrive.orchestration.parser import CanonicalSpec


def test_extract_slab_bounds_middle_layer():
    spec = CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dielectric_interfaces",
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
            ],
            "charges": [],
        }
    )
    bounds = _extract_slab_bounds(spec)
    assert bounds == (-0.3, 0.0)
