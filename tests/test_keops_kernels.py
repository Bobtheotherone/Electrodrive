import math
import os
import sys

import pytest
import torch

from electrodrive.core.bem_kernel import (
    bem_matvec_gpu,
    bem_potential_targets,
    bem_E_field_targets,
    _bem_matvec_core_torch,
    _bem_potential_targets_core_torch,
    _bem_E_field_targets_core_torch,
)
from electrodrive.core.bem_quadrature import self_integral_correction
from electrodrive.utils.config import K_E

# Try importing KeOps; skip tests if unavailable.
_IS_WINDOWS = sys.platform.startswith("win")
_KEOPS_IMPORT_ERROR = None
LazyTensor = None  # type: ignore
HAVE_KEOPS = False

if not _IS_WINDOWS:
    try:  # pragma: no cover - environment dependent
        from pykeops.torch import LazyTensor  # type: ignore

        HAVE_KEOPS = True
    except Exception as exc:  # pragma: no cover
        _KEOPS_IMPORT_ERROR = exc
        HAVE_KEOPS = False

if _IS_WINDOWS:
    _KEOPS_REASON = (
        "PyKeOps is not supported on Windows (requires a POSIX C++ toolchain)"
    )
else:
    _KEOPS_REASON = (
        f"pykeops is required for KeOps kernel tests ({_KEOPS_IMPORT_ERROR})"
        if not HAVE_KEOPS and _KEOPS_IMPORT_ERROR is not None
        else "pykeops is required for KeOps kernel tests"
    )

pytestmark = pytest.mark.skipif(
    _IS_WINDOWS or not HAVE_KEOPS,
    reason=_KEOPS_REASON,
)

def _tiny_plane_mesh(n: int = 6, L: float = 1.0, z: float = 0.0, dtype=torch.float64, device="cpu"):
    xs = torch.linspace(-L/2, L/2, n+1, dtype=dtype, device=device)
    ys = torch.linspace(-L/2, L/2, n+1, dtype=dtype, device=device)
    xv, yv = torch.meshgrid(xs, ys, indexing="ij")
    cx = 0.5 * (xv[:-1, :-1] + xv[1:, 1:])
    cy = 0.5 * (yv[:-1, :-1] + yv[1:, 1:])
    cz = torch.full_like(cx, z)
    centroids = torch.stack([cx.reshape(-1), cy.reshape(-1), cz.reshape(-1)], dim=1)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    areas = torch.full((centroids.shape[0],), float(dx * dy), dtype=dtype, device=device)
    normals = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).repeat(centroids.shape[0], 1)
    return centroids, areas, normals

def _self_integrals_from_area(areas: torch.Tensor) -> torch.Tensor:
    # Use the same function as production to avoid drift.
    out = torch.empty_like(areas)
    for i in range(areas.numel()):
        out[i] = self_integral_correction(areas[i])
    return out

def test_keops_vs_torch_matvec_double_precision():
    dtype = torch.float64
    device = torch.device("cpu")
    # Use enough panels to exceed the bem_matvec_gpu KeOps threshold (N>=2048).
    # This keeps the mesh simple but guarantees the KeOps backend is exercised.
    C, A, _ = _tiny_plane_mesh(n=46, dtype=dtype, device=device)
    N = C.shape[0]
    sigma = torch.linspace(0.1, 0.9, N, device=device, dtype=dtype)
    self_int = _self_integrals_from_area(A)

    V_ref = _bem_matvec_core_torch(
        centroids=C,
        areas=A,
        sigma=sigma,
        self_integrals=self_int,
        tile_size=64,
    )
    V_keops = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=C,
        areas=A,
        tile_size=64,
        self_integrals=self_int,
        use_near_quad=True,
        use_keops=True,  # force keops
        backend="keops",
    )
    max_diff = torch.max(torch.abs(V_ref - V_keops)).item()
    assert max_diff < 1e-9

def test_keops_vs_torch_potential_targets_double_precision():
    dtype = torch.float64
    device = torch.device("cpu")
    C, A, _ = _tiny_plane_mesh(n=5, dtype=dtype, device=device)
    N = C.shape[0]
    sigma = torch.cos(torch.linspace(0.0, 1.0, N, device=device, dtype=dtype))
    P = torch.tensor([[0.0, 0.0, 0.25], [0.1, -0.2, 0.5]], device=device, dtype=dtype)

    V_ref = _bem_potential_targets_core_torch(
        targets=P,
        src_centroids=C,
        areas=A,
        sigma=sigma,
        tile_size=64,
    )
    V_keops = bem_potential_targets(
        targets=P,
        src_centroids=C,
        areas=A,
        sigma=sigma,
        tile_size=64,
        use_keops=True,
    )

    max_diff = torch.max(torch.abs(V_ref - V_keops)).item()
    assert max_diff < 1e-9

def test_keops_vs_torch_Efield_targets_double_precision():
    dtype = torch.float64
    device = torch.device("cpu")
    C, A, _ = _tiny_plane_mesh(n=5, dtype=dtype, device=device)
    N = C.shape[0]
    sigma = torch.sin(torch.linspace(0.0, 2.0, N, device=device, dtype=dtype))
    P = torch.tensor([[0.0, 0.0, 0.3], [0.2, 0.1, 0.6]], device=device, dtype=dtype)

    E_ref = _bem_E_field_targets_core_torch(
        targets=P,
        src_centroids=C,
        areas=A,
        sigma=sigma,
        tile_size=64,
    )
    E_keops = bem_E_field_targets(
        targets=P,
        src_centroids=C,
        areas=A,
        sigma=sigma,
        tile_size=64,
        use_keops=True,
    )

    max_diff = torch.max(torch.abs(E_ref - E_keops)).item()
    assert max_diff < 1e-9
