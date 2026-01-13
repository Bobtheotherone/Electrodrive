import pytest
import torch

from electrodrive.gfdsl.eval.kernels_complex import complex_conjugate_pair_columns
from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import ImageSystem


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for complex pair component test"
)
def test_complex_pair_components_match_gfdsl_cuda() -> None:
    device = torch.device("cuda")
    dtype = torch.float32

    pos = torch.tensor([0.2, -0.1, 0.3], device=device, dtype=dtype)
    z_imag = torch.tensor(0.4, device=device, dtype=dtype)
    elem_real = PointChargeBasis(
        {"position": pos, "z_imag": z_imag, "component": torch.tensor(0.0, device=device, dtype=dtype)}
    )
    elem_imag = PointChargeBasis(
        {"position": pos, "z_imag": z_imag, "component": torch.tensor(1.0, device=device, dtype=dtype)}
    )

    targets = torch.tensor(
        [[0.0, 0.1, 0.2], [0.3, -0.2, 0.5], [-0.1, 0.4, -0.3], [0.2, 0.2, 0.7]],
        device=device,
        dtype=dtype,
    )

    phi_real = elem_real.potential(targets)
    phi_imag = elem_imag.potential(targets)

    xyab = torch.stack([pos[0], pos[1], pos[2], z_imag]).view(1, 4)
    cols = complex_conjugate_pair_columns(targets, xyab)

    assert torch.allclose(phi_real, cols[:, 0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(phi_imag, cols[:, 1], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for complex pair ImageSystem test"
)
def test_complex_pair_imagesystem_cuda_real_output() -> None:
    device = torch.device("cuda")
    dtype = torch.float32

    pos = torch.tensor([0.1, 0.2, 0.4], device=device, dtype=dtype)
    z_imag = torch.tensor(0.25, device=device, dtype=dtype)
    elems = [
        PointChargeBasis(
            {"position": pos, "z_imag": z_imag, "component": torch.tensor(0.0, device=device, dtype=dtype)}
        ),
        PointChargeBasis(
            {"position": pos, "z_imag": z_imag, "component": torch.tensor(1.0, device=device, dtype=dtype)}
        ),
    ]
    weights = torch.tensor([0.7, -0.4], device=device, dtype=dtype)
    system = ImageSystem(elems, weights)

    targets = torch.tensor(
        [[0.0, 0.0, 0.3], [0.2, -0.1, 0.5], [-0.2, 0.3, -0.1]],
        device=device,
        dtype=dtype,
    )
    V = system.potential(targets)

    assert V.is_cuda
    assert torch.isfinite(V).all()
    assert not torch.is_complex(V)
