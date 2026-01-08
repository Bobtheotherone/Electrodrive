import pytest
import torch

from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.image_system_v2 import ImageSystemV2
from electrodrive.images.search import ImageSystem


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for ImageSystemV2 tests"
)


def test_image_system_v2_matches_baseline():
    device = torch.device("cuda")
    elements = []
    for z in (0.2, 0.4, 0.6, 0.8):
        elem = PointChargeBasis({"position": torch.tensor([0.1, -0.2, z], device=device)})
        elements.append(elem)
    weights = torch.tensor([1.0, -0.5, 0.25, -0.75], device=device)

    sys = ImageSystem(elements, weights)
    sys_v2 = ImageSystemV2(elements, weights)

    pts = torch.randn(64, 3, device=device)
    V_ref = sys.potential(pts)
    V_v2 = sys_v2.potential(pts)

    assert V_v2.is_cuda
    assert torch.allclose(V_ref, V_v2, rtol=1e-5, atol=1e-6)
    assert sys_v2._point_real_pos is not None
    assert int(sys_v2._point_real_pos.shape[0]) == len(elements)
