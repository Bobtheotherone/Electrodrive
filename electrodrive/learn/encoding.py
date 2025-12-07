# electrodrive/learn/encoding.py
import torch
from typing import Dict

from electrodrive.orchestration.parser import CanonicalSpec

# Fixed-size encoding for conditional models.

MAX_SOURCES = 4
SOURCE_DIM = 6  # (q/lambda, x, y, z, type_point, type_line)
GEOMETRY_TYPE_DIM = 6  # unknown, plane, sphere, cylinder, parallel_planes, wedge
GEOMETRY_PARAM_DIM = 8
ENCODING_DIM = (
    MAX_SOURCES * SOURCE_DIM
    + GEOMETRY_TYPE_DIM
    + GEOMETRY_PARAM_DIM
)

GEOMETRY_TYPES = {
    "unknown": 0,
    "plane": 1,
    "sphere": 2,
    "cylinder": 3,
    "parallel_planes": 4,
    "wedge": 5,
}


def encode_spec(spec: CanonicalSpec) -> torch.Tensor:
    """Encode a CanonicalSpec into a fixed-size tensor [ENCODING_DIM].

    - Accepts both 'line' and 'line_charge' for line-like sources.
    - Infers geometry type heuristically from conductors; does NOT depend on
      any on-disk schema changes.
    """
    # 1) Sources
    source_features = torch.zeros(
        MAX_SOURCES * SOURCE_DIM,
        dtype=torch.float32,
    )
    count = 0
    for charge in spec.charges:
        if count >= MAX_SOURCES:
            break
        base = count * SOURCE_DIM
        ctype = charge.get("type")
        if ctype == "point":
            source_features[base] = float(
                charge.get("q", 0.0)
            )
            pos = charge.get(
                "pos",
                [0.0, 0.0, 0.0],
            )
            source_features[
                base
                + 1 : base
                + 4
            ] = torch.tensor(
                pos[:3],
                dtype=torch.float32,
            )
            source_features[
                base + 4
            ] = 1.0
            count += 1
        elif ctype in (
            "line_charge",
            "line",
        ):
            source_features[base] = float(
                charge.get("lambda", 0.0)
            )
            pos_2d = charge.get("pos_2d")
            if pos_2d:
                source_features[
                    base + 1
                ] = float(pos_2d[0])
                source_features[
                    base + 2
                ] = float(pos_2d[1])
            source_features[
                base + 5
            ] = 1.0
            count += 1

    # 2) Geometry heuristic
    geom_type_features = torch.zeros(
        GEOMETRY_TYPE_DIM,
        dtype=torch.float32,
    )
    geom_param_features = torch.zeros(
        GEOMETRY_PARAM_DIM,
        dtype=torch.float32,
    )

    ctypes = (
        sorted(
            {
                c.get("type")
                for c in spec.conductors
            }
        )
        if spec.conductors
        else []
    )
    geom_type = "unknown"

    if ctypes == ["plane"]:
        if len(spec.conductors) == 1:
            geom_type = "plane"
            geom_param_features[
                0
            ] = float(
                spec.conductors[0].get(
                    "z", 0.0
                )
            )
        elif len(spec.conductors) == 2:
            geom_type = (
                "parallel_planes"
            )
            try:
                z1 = float(
                    spec.conductors[
                        0
                    ].get("z")
                )
                z2 = float(
                    spec.conductors[
                        1
                    ].get("z")
                )
                geom_param_features[
                    0
                ] = (
                    abs(z1 - z2)
                    / 2.0
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    elif (
        ctypes == ["sphere"]
        and len(spec.conductors)
        == 1
    ):
        geom_type = "sphere"
        c = spec.conductors[0]
        geom_param_features[
            0
        ] = float(
            c.get("radius", 1.0)
        )
        center = c.get(
            "center",
            [0.0, 0.0, 0.0],
        )
        geom_param_features[
            1:4
        ] = torch.tensor(
            center[:3],
            dtype=torch.float32,
        )

    elif (
        ("cylinder" in ctypes)
        or ("cylinder2D" in ctypes)
    ) and len(spec.conductors) == 1:
        geom_type = "cylinder"
        geom_param_features[
            0
        ] = float(
            spec.conductors[
                0
            ].get("radius", 1.0)
        )

    if geom_type in GEOMETRY_TYPES:
        geom_type_features[
            GEOMETRY_TYPES[
                geom_type
            ]
        ] = 1.0

    return torch.cat(
        [
            source_features,
            geom_type_features,
            geom_param_features,
        ]
    )