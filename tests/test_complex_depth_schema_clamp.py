import torch
import pytest

from electrodrive.flows.schemas import ComplexDepthPointSchema


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for schema transform")
def test_complex_depth_schema_clamps_imag_depth():
    schema = ComplexDepthPointSchema("complex_depth_point", 3, 4, (0, 1, 2, 3))
    device = torch.device("cuda")

    u = torch.zeros((2, 4), device=device)
    u[0, 3] = -100.0
    u[1, 3] = 100.0

    out = schema.transform(u, spec=None)
    z_imag = out["z_imag"].abs()

    min_val = float(z_imag.min().item())
    max_val = float(z_imag.max().item())
    assert min_val >= 1e-3
    assert max_val <= 8.0 + 1e-6
