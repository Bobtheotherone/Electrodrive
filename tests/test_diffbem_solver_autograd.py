import math

import pytest
import torch

from electrodrive.core.bem_mesh import generate_mesh
from electrodrive.core.bem_quadrature import self_integral_correction
from electrodrive.core.bem_kernel import _bem_matvec_core_torch
from electrodrive.core import diffbem
from electrodrive.utils.logging import JsonlLogger
from electrodrive.utils.config import BEMConfig
from electrodrive.orchestration.parser import CanonicalSpec

try:  # pragma: no cover
    import xitorch  # type: ignore
    HAVE_XITORCH = True
except Exception:  # pragma: no cover
    HAVE_XITORCH = False

pytestmark = pytest.mark.skipif(
    not HAVE_XITORCH,
    reason="xitorch is required for differentiable solver tests",
)

class _SilentLogger(JsonlLogger):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass

def _simple_spec():
    # Plane at z=0, grounded; one point charge (just to create a RHS-like scale)
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "Dirichlet",
            "conductors": [{"type": "plane", "z": 0.0, "potential": 0.0}],
            "dielectrics": [],
            "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.5]}],
        }
    )

@pytest.mark.parametrize("use_cuda", [False])
def test_solver_autograd_wrt_rhs_scale(tmp_path, use_cuda):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    dtype = torch.float64

    spec = _simple_spec()
    logger = _SilentLogger(tmp_path)

    cfg = BEMConfig(
        use_gpu=(device.type == "cuda"),
        fp64=True,
        # Keep the mesh modest to keep the test runtime manageable.
        initial_h=0.6,
        max_refine_passes=1,
        # Slightly relaxed GMRES settings: still exercises autograd but avoids
        # long runs / convergence warnings seen in CI and WSL.
        gmres_tol=1e-5,
        gmres_maxiter=80,
    )

    # Build one pass mesh and tensors similar to bem_solve() internals
    mesh = generate_mesh(spec, target_h=0.6, logger=logger)
    N = mesh.n_panels
    C = torch.as_tensor(mesh.centroids, device=device, dtype=dtype)
    A = torch.as_tensor(mesh.areas, device=device, dtype=dtype)
    Nrm = torch.as_tensor(mesh.normals, device=device, dtype=dtype)

    # Diagonal self-integrals
    self_corr = torch.empty(N, device=device, dtype=dtype)
    for i in range(N):
        self_corr[i] = self_integral_correction(A[i])

    # Differentiable matvec
    def matvec_sigma_diff(sig: torch.Tensor) -> torch.Tensor:
        return _bem_matvec_core_torch(
            centroids=C,
            areas=A,
            sigma=sig,
            self_integrals=self_corr,
            tile_size=64,
        )

    # Create a synthetic RHS that depends on a learnable scalar alpha
    torch.manual_seed(0)
    base_rhs = torch.randn(N, device=device, dtype=dtype)
    alpha = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
    rhs = alpha * base_rhs

    out = diffbem.solve_diffbem(
        spec=None,   # not needed for this test path
        cfg=cfg,
        logger=logger,
        C=C,
        N=Nrm,
        A=A,
        rhs=rhs,
        matvec=matvec_sigma_diff,
        x0=None,
    )
    sigma = out["sigma"]

    # Simple scalar loss that depends on sigma
    loss = (sigma ** 2).mean()
    loss.backward()

    assert alpha.grad is not None
    assert math.isfinite(float(alpha.grad.item()))

    logger.close()
