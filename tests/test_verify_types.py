import pytest
import torch

from electrodrive.verify.certificate import DiscoveryCertificate, canonical_hash
from electrodrive.verify.oracle_types import (
    CachePolicy,
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
    TraceLevel,
)
from electrodrive.verify.utils import sha256_json


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for verify tests")


def test_oracle_query_roundtrip() -> None:
    _skip_if_no_cuda()
    spec = {"kind": "unit", "param": 1}
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], device="cuda", dtype=torch.float32)
    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.BOTH,
        requested_fidelity=OracleFidelity.F1,
        device="cuda",
        dtype="float32",
        seed=123,
        budget={"time_ms": 1.0},
        cache_policy=CachePolicy.USE_CACHE,
        trace=TraceLevel.MINIMAL,
    )
    blob = query.to_json()
    query2 = OracleQuery.from_json(blob)
    assert query2.spec == query.spec
    assert query2.quantity == query.quantity
    assert query2.requested_fidelity == query.requested_fidelity
    assert query2.device.startswith("cuda")
    assert torch.allclose(query2.points, query.points)


def test_oracle_result_roundtrip() -> None:
    _skip_if_no_cuda()
    V = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    E = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
    valid_mask = torch.tensor([True, False], device="cuda", dtype=torch.bool)
    err = OracleErrorEstimate(type=ErrorEstimateType.HEURISTIC, metrics={"rel": 0.1}, confidence=0.9, notes=["ok"])
    cost = OracleCost(wall_ms=1.0, cuda_ms=0.5, peak_vram_mb=32.0)
    cache = OracleCacheStatus(status=CacheStatus.MISS, key="k", path=None)
    prov = OracleProvenance(
        git_sha="test",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        device_name=torch.cuda.get_device_name(0),
        device=str(V.device),
        dtype="float32",
        timestamp="2024-01-01T00:00:00Z",
    )
    result = OracleResult(
        V=V,
        E=E,
        valid_mask=valid_mask,
        method="stub",
        fidelity=OracleFidelity.F0,
        config_fingerprint="cfg",
        error_estimate=err,
        cost=cost,
        cache=cache,
        provenance=prov,
    )
    blob = result.to_json()
    result2 = OracleResult.from_json(blob)
    assert torch.allclose(result2.V, V)
    assert torch.allclose(result2.E, E)
    assert torch.equal(result2.valid_mask, valid_mask)
    assert result2.fidelity == result.fidelity


def test_certificate_hash_deterministic() -> None:
    cert = DiscoveryCertificate(
        spec_digest=sha256_json({"a": 1}),
        candidate_digest=sha256_json({"b": 2}),
        git_sha="test",
        hardware={"device": "cuda"},
        oracle_runs=[{"fidelity": "F0"}],
        gates={"A": {"status": "stub"}},
        final_status="stub",
        reasons=["not_implemented"],
        attachments=["log.json"],
    )
    hash1 = canonical_hash(cert.to_json())
    hash2 = canonical_hash(DiscoveryCertificate.from_json(cert.to_json()).to_json())
    assert hash1 == hash2
    assert cert.digest() == hash1


def test_cuda_enforcement() -> None:
    _skip_if_no_cuda()
    cpu_points = torch.zeros((1, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        OracleQuery(
            spec={"x": 1},
            points=cpu_points,
            quantity=OracleQuantity.POTENTIAL,
            requested_fidelity=OracleFidelity.F0,
            device="cuda",
            dtype="float32",
        )
    V = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    valid_mask_cpu = torch.tensor([True], dtype=torch.bool)
    err = OracleErrorEstimate(metrics={})
    cost = OracleCost(wall_ms=0.0, cuda_ms=0.0, peak_vram_mb=0.0)
    cache = OracleCacheStatus(status=CacheStatus.MISS)
    prov = OracleProvenance(
        git_sha="test",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        device_name=torch.cuda.get_device_name(0),
        device="cuda",
        dtype="float32",
        timestamp="2024-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        OracleResult(
            V=V,
            E=None,
            valid_mask=valid_mask_cpu,
            method="stub",
            fidelity=OracleFidelity.F0,
            config_fingerprint="cfg",
            error_estimate=err,
            cost=cost,
            cache=cache,
            provenance=prov,
        )
