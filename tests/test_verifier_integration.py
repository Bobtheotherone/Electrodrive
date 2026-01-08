import json
from pathlib import Path

import pytest
import torch

from electrodrive.verify import verifier as verifier_mod
from electrodrive.verify.gates import GateResult
from electrodrive.verify.verifier import VerificationPlan, Verifier
from electrodrive.verify.oracle_types import OracleFidelity
from electrodrive.verify.oracle_registry import OracleRegistry
from electrodrive.verify.oracle_backends import F0AnalyticOracleBackend, F1SommerfeldOracleBackend


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for verifier integration tests")


def _fast_registry() -> OracleRegistry:
    reg = OracleRegistry()
    reg.register(F0AnalyticOracleBackend())
    reg.register(F1SommerfeldOracleBackend())
    return reg


def test_verifier_plane_pass(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    plan = VerificationPlan()
    plan.thresholds["laplacian_linf"] = 2e-1
    plan.thresholds["slope_tol"] = 1.0
    plan.thresholds["bc_dirichlet"] = 1e-2
    plan.thresholds["stability"] = 0.2
    plan.thresholds["min_speedup"] = 0.1
    plan.samples.update({"A_interior": 32, "B_boundary": 32, "C_far": 32, "C_near": 32, "E_bench": 128})
    out_root = tmp_path / "runs"
    verifier = Verifier(registry=_fast_registry(), out_root=out_root)

    def _candidate_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)

    cert = verifier.run({"eval_fn": _candidate_fn}, spec, plan)
    assert cert.final_status in ("pass", "borderline")
    run_dirs = list(out_root.iterdir())
    assert run_dirs, "Verifier did not emit artifacts"
    cert_path = run_dirs[0] / "discovery_certificate.json"
    assert cert_path.exists()


def test_verifier_gate_a_exclusion_radius_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    plan = VerificationPlan()
    plan.gate_order = ["A"]
    plan.thresholds["laplacian_exclusion_radius"] = 0.123
    plan.samples["A_interior"] = 16

    captured = {}

    def _fake_gate(query, result, *, config=None):  # type: ignore[no-untyped-def]
        captured["exclusion_radius"] = config.get("exclusion_radius")
        return GateResult(gate="A", status="pass", metrics={}, thresholds={})

    monkeypatch.setattr(verifier_mod.gateA_pde, "run_gate", _fake_gate)

    out_root = tmp_path / "runs_exclusion"
    verifier = Verifier(registry=_fast_registry(), out_root=out_root)

    def _candidate_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)

    cert = verifier.run({"eval_fn": _candidate_fn}, spec, plan)
    assert cert.final_status == "pass"
    assert captured["exclusion_radius"] == pytest.approx(0.123)


def test_verifier_bad_candidate_fail(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}],
        "dielectrics": [],
    }
    bad_candidate = {"constant": 0.0}
    plan = VerificationPlan()
    plan.thresholds["slope_tol"] = 0.05
    plan.thresholds["laplacian_linf"] = 1e-3
    plan.thresholds["bc_dirichlet"] = 1e-4
    plan.samples.update({"A_interior": 32, "B_boundary": 32, "C_far": 32, "C_near": 32, "E_bench": 64})
    out_root = tmp_path / "runs_bad"
    verifier = Verifier(registry=_fast_registry(), out_root=out_root)
    cert = verifier.run(bad_candidate, spec, plan)
    assert cert.final_status in ("fail", "borderline")


def test_verifier_layered_escalation(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "layered",
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.5]}],
        "dielectrics": [
            {"epsilon": 2.0, "z_min": 0.0, "z_max": 1.0},
            {"epsilon": 4.0, "z_min": -1.0, "z_max": 0.0},
        ],
    }
    plan = VerificationPlan()
    plan.allow_f2 = False
    plan.oracle_budget["allow_f1_auto"] = True
    plan.samples.update({"A_interior": 24, "B_boundary": 24, "C_far": 24, "C_near": 24, "E_bench": 64})
    plan.start_fidelity = OracleFidelity.AUTO
    out_root = tmp_path / "runs_layered"
    verifier = Verifier(registry=_fast_registry(), out_root=out_root)
    cert = verifier.run({}, spec, plan)
    history = cert.gates.get("history", [])
    assert isinstance(history, list)
    assert any(entry.get("fidelity") == "F1" for entry in history)
