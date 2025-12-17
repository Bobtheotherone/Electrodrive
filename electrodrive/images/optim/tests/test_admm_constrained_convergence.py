import torch
import pytest

from electrodrive.images.optim import ADMMConfig, ConstraintSpec, SparseSolveRequest
from electrodrive.images.optim.constrained_admm import admm_constrained_solve
from electrodrive.images.optim.bases.fourier_planar import PlanarFFTConstraintOp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ADMM constrained solve")
def test_admm_constrained_reduces_constraint_violation():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    h, w = 8, 8
    n = h * w
    A = torch.eye(n, device=device, dtype=dtype)
    g = torch.randn(n, device=device, dtype=dtype)

    ky, kx = 1, 2
    mode_indices = [(ky, kx), (-ky % h, -kx % w)]
    spec = ConstraintSpec(
        name="fft_zero_mode",
        kind="eq",
        eps=0.0,
        basis="planar_fft",
        params={"grid_shape": (h, w), "mode_indices": mode_indices},
    )

    op = PlanarFFTConstraintOp(
        grid_shape=(h, w),
        mode_indices=mode_indices,
        device=device,
        dtype=dtype,
    )
    initial_resid = op.apply((-g).reshape(-1))
    init_norm = torch.linalg.norm(initial_resid)

    req = SparseSolveRequest(
        A=A,
        X=None,
        g=g,
        is_boundary=None,
        lambda_l1=1e-4,
        lambda_group=0.0,
        group_ids=None,
        weight_prior=None,
        lambda_weight_prior=0.0,
        normalize_columns=True,
        col_norms=None,
        constraints=[spec],
        max_iter=50,
        tol=1e-4,
        warm_start=None,
        return_stats=True,
        dtype_policy=None,
    )
    cfg = ADMMConfig(rho=1.0, max_iter=40, tol=1e-4, w_update_iters=10)
    result = admm_constrained_solve(req, cfg)

    r_final = A @ result.w - g
    final_norm = torch.linalg.norm(op.apply(r_final.reshape(-1)))

    assert torch.isfinite(result.w).all()
    assert float(final_norm.item()) < float(init_norm.item())
