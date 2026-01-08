import torch

from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import ImageSystem


def test_imagesystem_float64_dtype_propagation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos = torch.tensor([0.1, -0.2, 0.3], device=device, dtype=torch.float64)
    elem = PointChargeBasis({"position": pos, "z_imag": torch.tensor(0.2, device=device, dtype=torch.float64)})
    weights = torch.tensor([1.0], device=device, dtype=torch.float32)
    system = ImageSystem([elem], weights)
    pts = torch.randn(16, 3, device=device, dtype=torch.float64, requires_grad=True)
    out = system.potential(pts)
    assert out.dtype == torch.float64
    assert out.requires_grad
    grad = torch.autograd.grad(out.sum(), pts, retain_graph=False)[0]
    assert grad is not None
