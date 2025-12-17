import torch
import pytest

from electrodrive.images.optim import ConstraintSpec, SparseSolveRequest, implicit_lasso_solve, refine_and_certify
from electrodrive.images.optim.diagnostics import constraint_residuals_from_specs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP64 certification test")
def test_fp64_certification():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 40, 20
    A = torch.randn(n, k, device=device, dtype=dtype)
    w_true = torch.zeros(k, device=device, dtype=dtype)
    w_true[:3] = torch.randn(3, device=device, dtype=dtype)
    g = A.matmul(w_true)

    constraints = [
        ConstraintSpec(
            name="collocation_all",
            kind="eq",
            weight=1.0,
            eps=0.0,
            basis="collocation",
            params={},
        )
    ]

    req = SparseSolveRequest(
        A=A,
        X=None,
        g=g,
        is_boundary=None,
        lambda_l1=1e-3,
        lambda_group=0.0,
        group_ids=None,
        weight_prior=None,
        lambda_weight_prior=0.0,
        normalize_columns=True,
        col_norms=None,
        constraints=constraints,
        max_iter=200,
        tol=1e-6,
        warm_start=None,
        return_stats=True,
        dtype_policy=None,
    )

    w32 = implicit_lasso_solve(req).w
    res32 = constraint_residuals_from_specs(A, w32, g, constraints)

    w64, cert = refine_and_certify(req, solver="implicit_lasso")
    res64 = cert.get("constraint_residuals", {})

    assert "collocation_all" in res64
    assert res64["collocation_all"] <= res32.get("collocation_all", res64["collocation_all"]) + 1e-4
    assert "kkt_residual" in cert
    assert torch.isfinite(w64).all()
