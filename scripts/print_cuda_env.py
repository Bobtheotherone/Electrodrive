#!/usr/bin/env python3
"""Print a concise CUDA environment report and a tiny matmul sanity check."""

from __future__ import annotations

import time

import torch


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _matmul_sanity(device: torch.device) -> dict[str, float]:
    # Single-run matmul timing to confirm GPU path without overbenchmarking.
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = a @ b
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        _ = a @ b
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"matmul_ms": float(elapsed_ms)}


def main() -> None:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    default_dtype = torch.get_default_dtype()
    cudnn_available = torch.backends.cudnn.is_available()
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32

    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {_format_bool(cuda_available)}")
    print(f"device_name: {device_name}")
    print(f"device_capability: {capability}")
    print(f"default_dtype: {default_dtype}")
    print(f"cudnn_available: {_format_bool(cudnn_available)}")
    print(f"tf32_matmul: {_format_bool(tf32_matmul)}")
    print(f"tf32_cudnn: {_format_bool(tf32_cudnn)}")

    device = torch.device("cuda" if cuda_available else "cpu")
    metrics = _matmul_sanity(device)
    print(f"matmul_ms: {metrics['matmul_ms']:.3f}")


if __name__ == "__main__":
    main()
