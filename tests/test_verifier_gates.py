from pathlib import Path

import pytest
import torch

from electrodrive.verify.gates import gateA_pde, gateB_bc, gateC_asymptotics, gateD_stability, gateE_speed
from electrodrive.verify.oracle_types import (
    CacheStatus,
    ErrorEstimateType,
    OracleCacheStatus,
    OracleCost,
    OracleErrorEstimate,
    OracleFidelity,
    OracleProvenance,
    OracleQuery,
    OracleQuantity,
    OracleResult,
)


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for verifier gate tests")


def _dummy_result(points: torch.Tensor, fidelity: OracleFidelity = OracleFidelity.F0) -> tuple[OracleQuery, OracleResult]:
    query = OracleQuery(
        spec={"domain": "halfspace", "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 1.0]}]},
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=fidelity,
        device=str(points.device),
        dtype=str(points.dtype).replace("torch.", ""),
    )
    prov = OracleProvenance(
        git_sha="test",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        device_name=torch.cuda.get_device_name(0),
        device=str(points.device),
        dtype=str(points.dtype).replace("torch.", ""),
        timestamp="2024-01-01T00:00:00Z",
    )
    result = OracleResult(
        V=torch.zeros(points.shape[0], device=points.device, dtype=points.dtype),
        E=None,
        valid_mask=torch.ones(points.shape[0], device=points.device, dtype=torch.bool),
        method="stub",
        fidelity=fidelity,
        config_fingerprint="cfg",
        error_estimate=OracleErrorEstimate(type=ErrorEstimateType.HEURISTIC, metrics={}, confidence=0.1, notes=[]),
        cost=OracleCost(wall_ms=0.1, cuda_ms=0.05, peak_vram_mb=1.0),
        cache=OracleCacheStatus(status=CacheStatus.MISS),
        provenance=prov,
    )
    return query, result


def test_gate_a_pde_harmonic_pass(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(64, 3, device="cuda", dtype=torch.float32)
    query, result = _dummy_result(pts)

    def harmonic(p: torch.Tensor) -> torch.Tensor:
        return p.sum(dim=1)

    out = gateA_pde.run_gate(
        query,
        result,
        config={
            "candidate_eval": harmonic,
            "artifact_dir": tmp_path / "gateA",
            "spec": query.spec,
            "n_interior": 64,
            "linf_tol": 1e-2,
        },
    )
    assert out.status in ("pass", "borderline")
    assert out.metrics["linf"] < 1e-1


def test_gate_a_pde_interface_band_linear_pass(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(64, 3, device="cuda", dtype=torch.float32)
    _, result = _dummy_result(pts)
    layered_spec = {
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
    }
    query = OracleQuery(
        spec=layered_spec,
        points=pts,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F0,
        device=str(pts.device),
        dtype=str(pts.dtype).replace("torch.", ""),
    )

    def linear(p: torch.Tensor) -> torch.Tensor:
        return p[:, 2]

    out = gateA_pde.run_gate(
        query,
        result,
        config={
            "candidate_eval": linear,
            "artifact_dir": tmp_path / "gateA_interface",
            "spec": layered_spec,
            "n_interior": 64,
            "linf_tol": 1e-2,
            "fd_h": 2e-2,
            "interface_band": 0.05,
            "prefer_autograd": False,
        },
    )
    assert out.status in ("pass", "borderline")
    assert out.metrics["linf"] < 1e-1


def test_gate_b_boundary_plane_pass(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(64, 3, device="cuda", dtype=torch.float32)
    query, result = _dummy_result(
        pts,
        fidelity=OracleFidelity.F0,
    )
    query.spec["conductors"] = [{"type": "plane", "z": 0.0, "potential": 0.0}]

    def candidate(p: torch.Tensor) -> torch.Tensor:
        return p[:, 2] * 0.0

    out = gateB_bc.run_gate(
        query,
        result,
        config={
            "eval_fn": candidate,
            "tolerance": 1e-3,
            "artifact_dir": tmp_path / "gateB",
        },
    )
    assert out.status == "pass"
    assert out.metrics["dirichlet_max_err"] <= 1e-3


def test_gate_c_asymptotics_point_charge(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(32, 3, device="cuda", dtype=torch.float32)
    query, result = _dummy_result(pts)

    def point_charge(p: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(p - torch.tensor([0.0, 0.0, 1.0], device=p.device, dtype=p.dtype), dim=1).clamp_min(1e-6)
        return 1.0 / r

    out = gateC_asymptotics.run_gate(
        query,
        result,
        config={
            "candidate_eval": point_charge,
            "artifact_dir": tmp_path / "gateC",
            "n_far": 32,
            "n_near": 32,
            "slope_tol": 0.5,
        },
    )
    assert out.status in ("pass", "borderline")
    assert abs(out.metrics["far_slope"] + 1.0) < 0.6


def test_gate_d_stability_constant(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(48, 3, device="cuda", dtype=torch.float32)
    query, result = _dummy_result(pts)

    def constant(p: torch.Tensor) -> torch.Tensor:
        return torch.ones(p.shape[0], device=p.device, dtype=p.dtype)

    out = gateD_stability.run_gate(
        query,
        result,
        config={"candidate_eval": constant, "delta": 1e-3, "artifact_dir": tmp_path / "gateD", "stability_tol": 1e-2},
    )
    assert out.status == "pass"
    assert out.metrics["relative_change"] <= 1e-2


def test_gate_e_speed_smoke(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    pts = torch.randn(128, 3, device="cuda", dtype=torch.float32)
    query, result = _dummy_result(pts)

    def fast_eval(p: torch.Tensor) -> torch.Tensor:
        return torch.sin(p[:, 0]) + p[:, 1] * 0.0

    def slow_eval(p: torch.Tensor) -> torch.Tensor:
        _ = torch.sin(p[:, 0]) + torch.cos(p[:, 1])
        torch.cuda.synchronize()
        return torch.sin(p[:, 0])

    out = gateE_speed.run_gate(
        query,
        result,
        config={
            "candidate_eval": fast_eval,
            "baseline_eval": slow_eval,
            "n_bench": 128,
            "min_speedup": 0.5,
            "artifact_dir": tmp_path / "gateE",
        },
    )
    assert out.status in ("pass", "borderline")
    assert out.metrics["samples"] == pytest.approx(128.0)
