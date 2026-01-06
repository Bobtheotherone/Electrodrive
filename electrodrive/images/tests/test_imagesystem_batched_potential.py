import pytest
import torch

from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import ImageSystem


def _make_elements(device: torch.device, dtype: torch.dtype) -> list[PointChargeBasis]:
    pos_base = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2], [0.1, 0.2, -0.6]],
        device=device,
        dtype=dtype,
    )
    elems = [
        PointChargeBasis({"position": pos_base[0]}),
        PointChargeBasis({"position": pos_base[1], "z_imag": torch.tensor(0.4, device=device, dtype=dtype)}),
        PointChargeBasis({"position": pos_base[2]}),
    ]
    return elems


def _reference_potential(
    elements: list[PointChargeBasis],
    weights: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    targets_sys = targets.to(device=device, dtype=dtype)
    V_ref = torch.zeros(targets_sys.shape[0], device=device, dtype=dtype)
    for elem, w in zip(elements, weights):
        V_ref.add_(w * elem.potential(targets_sys))
    return V_ref


def test_imagesystem_batched_potential_cpu() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    elements = _make_elements(device, dtype)
    weights = torch.tensor([0.6, -1.1, 0.25], device=device, dtype=dtype)
    system = ImageSystem(elements, weights)

    targets = torch.tensor(
        [[0.0, 0.1, 0.2], [0.3, -0.2, 0.5], [-0.1, 0.4, -0.3]],
        device=device,
        dtype=dtype,
    )

    V_ref = _reference_potential(elements, weights, targets, device, dtype)
    V_new = system.potential(targets)

    assert torch.allclose(V_new, V_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for batched ImageSystem test"
)
def test_imagesystem_batched_potential_cuda() -> None:
    dtype = torch.float32

    elements_cpu = _make_elements(torch.device("cpu"), dtype)
    weights_cpu = torch.tensor([0.6, -1.1, 0.25], device="cpu", dtype=dtype)
    system_cpu = ImageSystem(elements_cpu, weights_cpu)

    elements_gpu = _make_elements(torch.device("cuda"), dtype)
    weights_gpu = weights_cpu.to(device="cuda")
    system_gpu = ImageSystem(elements_gpu, weights_gpu)

    targets_cpu = torch.tensor(
        [[0.0, 0.1, 0.2], [0.3, -0.2, 0.5], [-0.1, 0.4, -0.3]],
        device="cpu",
        dtype=dtype,
    )
    targets_gpu = targets_cpu.to(device="cuda")

    V_cpu = system_cpu.potential(targets_cpu)
    V_gpu = system_gpu.potential(targets_gpu)

    assert V_gpu.is_cuda
    assert torch.isfinite(V_gpu).all()
    assert torch.allclose(V_gpu.cpu(), V_cpu, rtol=1e-4, atol=1e-5)
