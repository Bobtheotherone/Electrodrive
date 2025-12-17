"""GPU-first device utilities used throughout the GFlowNet stack.

The helpers here centralize device selection so new components default to CUDA
whenever available, while still remaining import-safe on CPU-only machines.
"""

from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Return the preferred device, prioritizing CUDA when available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_cuda_available_or_skip(test_context: str = "CUDA-required") -> None:
    """Skip a test when CUDA is unavailable.

    This helper should be used in GPU-first tests to avoid accidental CPU
    execution in CI environments that do not provide a CUDA runtime.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip(f"{test_context}: CUDA not available", allow_module_level=True)


def assert_cuda_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert that a tensor resides on CUDA in debug/test builds."""
    if not tensor.is_cuda:
        raise AssertionError(f"{name} expected to be on CUDA device, found {tensor.device}")
