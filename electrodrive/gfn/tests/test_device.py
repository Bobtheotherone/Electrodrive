from __future__ import annotations

import torch

from electrodrive.utils.device import assert_cuda_tensor, ensure_cuda_available_or_skip, get_default_device


def test_get_default_device_prefers_cuda() -> None:
    ensure_cuda_available_or_skip("get_default_device")
    device = get_default_device()
    assert device.type == "cuda"


def test_assert_cuda_tensor_accepts_cuda_tensor() -> None:
    ensure_cuda_available_or_skip("assert_cuda_tensor")
    tensor = torch.zeros(1, device="cuda")
    assert_cuda_tensor(tensor, name="probe")
