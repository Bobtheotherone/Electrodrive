import pytest
import torch

from electrodrive.verify.poles import find_poles_denominator_roots, find_poles_rational_fit


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pole engine tests")


def test_pole_engines_recover_synthetic_poles() -> None:
    _skip_if_no_cuda()
    device = torch.device("cuda")
    dtype = torch.complex128
    poles = [1.0 - 0.3j, 2.2 + 0.0j]
    residues = [0.7 + 0.1j, -0.4 + 0.05j]

    def numer_denom(k: torch.Tensor):
        k = k.to(device=device, dtype=dtype)
        denom = (k - poles[0]) * (k - poles[1])
        numer = residues[0] * (k - poles[1]) + residues[1] * (k - poles[0])
        return denom, numer

    def reflection_fn(k: torch.Tensor):
        d, n = numer_denom(k)
        return n / d

    k_samples = torch.linspace(0.05, 6.0, 256, device=device, dtype=dtype)

    poles_den = find_poles_denominator_roots(
        numer_denom,
        k_samples=k_samples,
        max_poles=4,
        device=device,
        dtype=dtype,
    )
    assert len(poles_den) >= 2
    recovered = sorted([p.pole for p in poles_den], key=lambda x: x.real)
    target = sorted(poles, key=lambda x: x.real)
    for r, t in zip(recovered[:2], target):
        assert abs(r - t) < 1e-3

    poles_fit = find_poles_rational_fit(
        reflection_fn,
        k_samples=k_samples,
        max_poles=4,
        device=device,
        dtype=dtype,
    )
    assert poles_fit
    recovered_fit = sorted([p.pole for p in poles_fit], key=lambda x: x.real)
    for r, t in zip(recovered_fit[:2], target):
        assert abs(r - t) < 5e-2
