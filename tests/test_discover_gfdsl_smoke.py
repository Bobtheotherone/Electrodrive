import json
from pathlib import Path

import pytest
import torch

from electrodrive.gfdsl.ast import Param, RealImageChargeNode
from electrodrive.gfdsl.io import serialize_program
from electrodrive.images.search import discover_images
from electrodrive.orchestration.parser import parse_spec
from electrodrive.utils.logging import JsonlLogger


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GFDSL discovery smoke test"
)


def test_discover_gfdsl_smoke(tmp_path: Path):
    device = torch.device("cuda")
    program_dir = tmp_path / "gfdsl_programs"
    program_dir.mkdir(parents=True, exist_ok=True)

    program = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, -0.25], device=device)),
        }
    )
    program_payload = serialize_program(program)
    program_path = program_dir / "prog0.json"
    program_path.write_text(json.dumps(program_payload, indent=2), encoding="utf-8")

    raw_spec = json.loads(Path("specs/plane_point_tiny.json").read_text(encoding="utf-8-sig"))
    spec = parse_spec(raw_spec)
    logger = JsonlLogger(tmp_path)

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=4,
        reg_l1=1e-6,
        restarts=1,
        logger=logger,
        basis_generator_mode="gfdsl",
        gfdsl_program_dir=str(program_dir),
        n_points_override=64,
        ratio_boundary_override=0.5,
    )

    assert system.weights.is_cuda
    assert len(system.elements) >= 1
