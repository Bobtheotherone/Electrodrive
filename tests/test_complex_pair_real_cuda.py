import torch

from electrodrive.images.basis import PointChargeBasis
from electrodrive.utils.config import K_E
from electrodrive.utils.device import ensure_cuda_available_or_skip


def test_complex_pair_real_cuda():
    ensure_cuda_available_or_skip("complex pair real-valuedness")
    device = torch.device("cuda")
    dtype = torch.float32
    pos = torch.tensor([0.1, -0.2, 0.3], device=device, dtype=dtype)
    z_imag = torch.tensor(0.4, device=device, dtype=dtype)
    elem = PointChargeBasis({"position": pos, "z_imag": z_imag})

    targets = torch.randn(16, 3, device=device, dtype=dtype)
    out = elem.potential(targets)
    assert out.is_cuda
    assert out.dtype == dtype
    assert not torch.is_complex(out)

    delta = targets - pos
    dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
    dz_complex = torch.complex(dz, z_imag.expand_as(dz))
    r2 = dx * dx + dy * dy + dz_complex * dz_complex
    expected = K_E * (2.0 / torch.sqrt(r2)).real
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5)
