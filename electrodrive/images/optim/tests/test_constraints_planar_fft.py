import torch
import pytest

from electrodrive.images.optim.bases.fourier_planar import PlanarFFTConstraintOp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for planar FFT constraints")
def test_planar_fft_constraint_apply_adjoint():
    device = torch.device("cuda")
    dtype = torch.float32
    h, w = 16, 16
    ky, kx = 3, 2

    f = torch.zeros((h, w), device=device, dtype=torch.complex64)
    f[ky, kx] = 1.0 + 0.0j
    f[-ky % h, -kx % w] = 1.0 + 0.0j
    r = torch.fft.ifft2(f).real

    op = PlanarFFTConstraintOp(
        grid_shape=(h, w),
        mode_indices=[(ky, kx), (-ky % h, -kx % w)],
        device=device,
        dtype=dtype,
    )
    coeffs = op.apply(r.reshape(-1))
    n_modes = 2
    coeffs_complex = coeffs[:n_modes] + 1j * coeffs[n_modes:]

    assert torch.allclose(
        coeffs_complex.abs(),
        torch.ones_like(coeffs_complex.abs()),
        atol=1e-2,
        rtol=1e-2,
    )

    r_back = op.adjoint(coeffs)
    r_flat = r.reshape(-1)
    corr = torch.dot(r_back, r_flat) / (
        torch.linalg.norm(r_back) * torch.linalg.norm(r_flat) + 1e-9
    )
    assert float(corr.item()) > 0.9
