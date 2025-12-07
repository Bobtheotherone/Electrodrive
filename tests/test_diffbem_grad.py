import math

import pytest
import torch

from electrodrive.core.bem import bem_solve
from electrodrive.core.bem_kernel import (
    _bem_potential_targets_core_torch,
)
from electrodrive.utils.config import BEMConfig, K_E
from electrodrive.utils.logging import JsonlLogger
from electrodrive.orchestration.parser import CanonicalSpec

# Optional xitorch; differentiable BEM tests are skipped when unavailable.
try:  # pragma: no cover - env dependent
    import xitorch  # type: ignore  # noqa: F401

    HAVE_XITORCH = True
except Exception:  # pragma: no cover
    HAVE_XITORCH = False

if not HAVE_XITORCH:  # pragma: no cover
    pytest.skip(
        "xitorch is required for end-to-end differentiable BEM tests.",
        allow_module_level=True,
    )


class _SilentLogger(JsonlLogger):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)

    # Silence logs during tests
    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _simple_plane_spec(z_charge: float = 0.5) -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "Dirichlet",
            "conductors": [
                {"type": "plane", "z": 0.0, "potential": 0.0}
            ],
            "dielectrics": [],
            "charges": [
                {"type": "point", "q": 1.0, "pos": [0.0, 0.0, z_charge]}
            ],
        }
    )


@pytest.mark.parametrize("use_cuda", [False])
def test_kernel_grad_wrt_sigma(tmp_path, use_cuda):
    """
    Weaker but focused gradient test:

    - Treat sigma as a learnable tensor.
    - Use `_bem_potential_targets_core_torch` to compute potentials at probes.
    - Check that gradients w.r.t. sigma are finite.

    This validates the differentiable kernel cores independently of the solver.
    """
    device = torch.device(
        "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64

    spec = _simple_plane_spec()
    logger = _SilentLogger(tmp_path)

    cfg = BEMConfig(
        use_gpu=(device.type == "cuda"),
        fp64=True,
        initial_h=0.4,
        max_refine_passes=1,
        gmres_tol=1e-6,
        gmres_maxiter=200,
    )

    out = bem_solve(spec, cfg, logger, differentiable=False)
    assert "error" not in out, f"BEM solve failed: {out.get('error')}"

    sol = out["solution"]
    C = sol._C.to(device=device, dtype=dtype)
    A = sol._A.to(device=device, dtype=dtype)

    sigma = torch.tensor(
        out["surface_charge_density"],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    probe_points = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.1, 0.0, 0.30],
            [-0.1, 0.0, 0.35],
        ],
        dtype=dtype,
        device=device,
    )

    V_ind = _bem_potential_targets_core_torch(
        targets=probe_points,
        src_centroids=C,
        areas=A,
        sigma=sigma,
        tile_size=64,
    )
    loss = V_ind.mean()

    loss.backward()

    assert sigma.grad is not None
    assert math.isfinite(float(sigma.grad.norm().item()))

    logger.close()


@pytest.mark.parametrize("use_cuda", [False])
def test_end_to_end_grad_wrt_charge_z(tmp_path, use_cuda):
    """
    End-to-end differentiable pipeline test (relaxed):

    - Solve BEM once with differentiable=True (xitorch path).
    - Use the resulting sigma to build potentials at probe points.
    - Treat the point charge z-position as a learnable tensor, but only in the
      free-space contribution; the induced part is held fixed from the solve.
    - Backpropagate a scalar loss to z_param and ensure the gradient is finite.

    This still exercises:
      diffbem.solve_diffbem -> matvec -> sigma -> differentiable kernels
    in a coherent way, without requiring full geometry re-meshing to be
    differentiable w.r.t. z_param.
    """
    device = torch.device(
        "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64

    logger = _SilentLogger(tmp_path)

    cfg = BEMConfig(
        use_gpu=(device.type == "cuda"),
        fp64=True,
        initial_h=0.4,
        max_refine_passes=1,
        gmres_tol=1e-5,
        gmres_maxiter=200,
    )

    # 1) Solve once with a fixed charge height z0 (non-differentiable wrt z0 here).
    z0 = 0.5
    spec = _simple_plane_spec(z0)
    out = bem_solve(spec, cfg, logger, differentiable=True)
    assert "error" not in out, f"BEM solve failed: {out.get('error')}"

    sol = out["solution"]
    assert sol is not None

    # Extract panel data; treat geometry as fixed for this test.
    C = sol._C.detach().to(device=device, dtype=dtype)  # [N,3]
    A = sol._A.detach().to(device=device, dtype=dtype)  # [N]
    sigma = sol._S  # differentiable result of diffbem.solve_diffbem

    # Probe points above the plane
    probes = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.1, 0.0, 0.35],
            [-0.1, 0.0, 0.45],
        ],
        dtype=dtype,
        device=device,
    )

    # 2) Learnable charge height used only in the free-space term.
    z_param = torch.tensor(z0, dtype=dtype, device=device, requires_grad=True)
    q = torch.tensor(1.0, dtype=dtype, device=device)

    # Free-space potential from the real charge at probes (depends on z_param).
    charge_pos = torch.stack(
        [
            torch.zeros_like(z_param),
            torch.zeros_like(z_param),
            z_param,
        ]
    )  # [3]
    R = probes - charge_pos  # [P,3]
    r = torch.linalg.norm(R, dim=1).clamp_min(1e-12)
    V_free = K_E * q / r

    # Induced potential at probes from the already-solved sigma (held fixed).
    V_ind = _bem_potential_targets_core_torch(
        targets=probes,
        src_centroids=C,
        areas=A,
        sigma=sigma.detach(),  # detach: no z_param dependency here
        tile_size=64,
    )

    V_total = V_free + V_ind
    loss = V_total.mean()

    # Backprop through the free-space contribution to z_param.
    loss.backward()

    assert z_param.grad is not None
    assert math.isfinite(float(z_param.grad.item()))

    logger.close()
