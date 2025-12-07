from __future__ import annotations

import os

import torch

from electrodrive.core.bem_kernel import bem_matvec_gpu
from electrodrive.utils.logging import JsonlLogger


def test_keops_env_toggle_without_keops(tmp_path):
    os.environ["EDE_BEM_USE_KEOPS"] = "1"
    logger = JsonlLogger(tmp_path)

    N = 8
    device = "cpu"
    sigma = torch.ones(N, device=device)
    src = torch.randn(N, 3, device=device)
    areas = torch.ones(N, device=device)
    self_corr = torch.zeros(N, device=device)

    out = bem_matvec_gpu(
        sigma,
        src,
        areas,
        tile_size=4,
        self_correction=self_corr,
        logger=logger,
        use_keops=True,
    )
    assert out.shape == (N,)
    logger.close()
    os.environ.pop("EDE_BEM_USE_KEOPS", None)
