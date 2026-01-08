import pytest
import torch


def test_cuda_smoke_matmul():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    capability = torch.cuda.get_device_capability(0)
    assert capability is not None
    assert capability >= (8, 0)

    torch.manual_seed(0)
    a_cpu = torch.randn(256, 256)
    b_cpu = torch.randn(256, 256)
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()

    c_gpu = a_gpu @ b_gpu
    assert c_gpu.is_cuda
    assert torch.isfinite(c_gpu).all()

    c_cpu = a_cpu @ b_cpu
    c_cpu_gpu = c_cpu.cuda()

    assert torch.allclose(c_gpu, c_cpu_gpu, rtol=1e-3, atol=1e-4)
