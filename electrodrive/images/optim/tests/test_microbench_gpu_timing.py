import math

import torch
import pytest

from electrodrive.images.search import solve_sparse
from electrodrive.images.optim import ADMMConfig
from electrodrive.images.optim.diagnostics import CudaTimer
from electrodrive.utils.logging import JsonlLogger


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU timing microbench")
def test_microbench_gpu_timing(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    n, k = 48, 24
    A = torch.randn(n, k, device=device, dtype=dtype)
    w_true = torch.randn(k, device=device, dtype=dtype)
    g = A @ w_true
    X = torch.empty((0, 3), device=device, dtype=dtype)

    timings = {}
    with JsonlLogger(tmp_path) as logger:
        for solver in ("ista", "implicit_lasso", "admm_constrained"):
            cfg = None
            if solver == "admm_constrained":
                cfg = ADMMConfig(max_iter=20, w_update_iters=5)
            timer = CudaTimer()
            with timer:
                solve_sparse(
                    A,
                    X,
                    g,
                    None,
                    logger,
                    reg_l1=1e-3,
                    solver=solver,
                    max_iter=20,
                    tol=1e-4,
                    constraint_mode="none",
                    admm_cfg=cfg,
                    return_stats=False,
                )
            timings[solver] = timer.elapsed_ms

    for value in timings.values():
        assert math.isfinite(value)
        assert value >= 0.0
