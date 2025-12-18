from pathlib import Path

import numpy as np
import pytest
import torch

from tools.verify_discovery import _load_points


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for verify_discovery point loading tests")


def test_load_points_from_npz(tmp_path: Path) -> None:
    _skip_if_no_cuda()
    arr = np.array([[0.0, 0.1, 0.2], [1.0, -0.5, 0.3]], dtype=np.float32)
    npz_path = tmp_path / "points.npz"
    np.savez(npz_path, points=arr)

    pts = _load_points(npz_path, device=torch.device("cuda"), dtype=torch.float32)

    assert pts.shape == (2, 3)
    assert pts.device.type == "cuda"
    assert torch.allclose(pts.cpu(), torch.from_numpy(arr))
