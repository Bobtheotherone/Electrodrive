import json
from pathlib import Path

import pytest
import torch

from electrodrive.gfdsl.ast import Param, RealImageChargeNode
from electrodrive.gfdsl.io import serialize_program
from electrodrive.tools.gfdsl_verify import run_gfdsl_verify
from electrodrive.verify.verifier import VerificationPlan


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GFDSL runway smoke test"
)


def test_gfdsl_runway_smoke(tmp_path: Path):
    device = torch.device("cuda")
    program = RealImageChargeNode(
        params={
            "position": Param(torch.tensor([0.0, 0.0, 0.25], device=device)),
        }
    )

    program_path = tmp_path / "program.json"
    program_payload = serialize_program(program)
    program_path.write_text(json.dumps(program_payload, indent=2), encoding="utf-8")

    spec_path = Path("specs/plane_point_tiny.json")
    out_dir = tmp_path / "run"

    plan = VerificationPlan()
    plan.gate_order = ["A"]
    plan.samples["A_interior"] = 16

    artifacts, certificate = run_gfdsl_verify(
        spec_path=spec_path,
        program_path=program_path,
        out_dir=out_dir,
        dtype=torch.float32,
        eval_backend="operator",
        solver="ista",
        reg_l1=1e-4,
        seed=123,
        gates=["A"],
        n_points=64,
        max_iter=200,
        plan=plan,
    )

    assert artifacts.program_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.weights_path.exists()
    assert artifacts.certificate_path.exists()
    assert "A" in certificate.gates
