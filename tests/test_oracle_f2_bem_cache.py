from pathlib import Path

import pytest
import torch

from electrodrive.verify.oracle_backends import F2BEMOracleBackend
from electrodrive.verify.oracle_types import CachePolicy, CacheStatus, OracleFidelity, OracleQuery, OracleQuantity


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for BEM oracle tests")


def test_bem_cache_hit(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 0.5, "pos": [0.0, 0.0, 0.8]}],
        "dielectrics": [],
    }
    points = torch.tensor([[0.0, 0.0, 0.2], [0.1, 0.0, 0.3]], device="cuda", dtype=torch.float64)
    backend = F2BEMOracleBackend(disk_root=tmp_path / "bem_cache")
    budget = {"bem": {"max_refine_passes": 1, "gmres_tol": 1e-3, "target_bc_inf_norm": 1e-3}}

    query = OracleQuery(
        spec=spec,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F2,
        cache_policy=CachePolicy.USE_CACHE,
        budget=budget,
    )

    res1 = backend.evaluate(query)
    assert res1.cache.status in (CacheStatus.MISS, CacheStatus.HIT)

    res2 = backend.evaluate(query)
    assert res2.cache.status == CacheStatus.HIT
    assert res2.cache.key == res1.cache.key


def test_bem_ram_cache_key_includes_spec(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    spec_a = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 0.5, "pos": [0.0, 0.0, 0.8]}],
        "dielectrics": [],
    }
    spec_b = {
        "domain": "halfspace",
        "BCs": "Dirichlet",
        "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.8]}],
        "dielectrics": [],
    }
    points = torch.tensor([[0.0, 0.0, 0.25], [0.15, -0.05, 0.35]], device="cuda", dtype=torch.float64)
    backend = F2BEMOracleBackend(disk_root=tmp_path / "bem_cache_ram")
    budget = {"bem": {"max_refine_passes": 1, "gmres_tol": 1e-3, "target_bc_inf_norm": 1e-3}}

    query_a = OracleQuery(
        spec=spec_a,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F2,
        cache_policy=CachePolicy.USE_CACHE,
        budget=budget,
    )
    res_a = backend.evaluate(query_a)

    query_b = OracleQuery(
        spec=spec_b,
        points=points,
        quantity=OracleQuantity.POTENTIAL,
        requested_fidelity=OracleFidelity.F2,
        cache_policy=CachePolicy.USE_CACHE,
        budget=budget,
    )
    res_b = backend.evaluate(query_b)

    assert res_b.cache.status == CacheStatus.MISS
    assert res_a.V is not None and res_b.V is not None
    assert not torch.allclose(res_a.V, res_b.V, atol=1e-4, rtol=1e-3)
