"""
Unit tests for the CUDA-accelerated near-field quadrature matvec.

This checks that the CUDA implementation in `electrodrive.core.bem_near_cuda`
produces the same result as the existing CPU-only helper
`_apply_near_quadrature_matvec` in `electrodrive.core.bem` on a random
synthetic mesh.

If CUDA or the extension is unavailable, the test is skipped.
"""

import numpy as np
import pytest
import torch

from electrodrive.core.bem import (
    _apply_near_quadrature_matvec,
    _apply_near_quadrature_matvec_cuda,
    _build_near_pairs_for_panels,
)
from electrodrive.core.bem_kernel import bem_matvec_gpu
from electrodrive.core.bem_quadrature import self_integral_correction
from electrodrive.core import bem_near_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_near_quadrature_matvec_cuda_matches_cpu():
    # Skip if the CUDA extension cannot be built / loaded.
    available = bem_near_cuda.is_bem_near_cuda_available()
    if not available:
        reason = "bem_near_cuda extension is not available"
        err = bem_near_cuda.get_bem_near_cuda_error()
        if err:
            reason = f"{reason}: {err}"
        pytest.skip(reason)

    device = torch.device("cuda")
    dtype = torch.float64

    # Small random mesh: N small triangles clustered near the origin
    N = 32
    rng = np.random.default_rng(1234)
    panel_vertices_np = 0.1 * rng.standard_normal(size=(N, 3, 3))

    # Compute areas and centroids on CPU
    v0 = panel_vertices_np[:, 0, :]
    v1 = panel_vertices_np[:, 1, :]
    v2 = panel_vertices_np[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas_np = 0.5 * np.linalg.norm(cross, axis=1)
    centroids_np = panel_vertices_np.mean(axis=1)

    # Drop degenerate triangles if any
    mask = areas_np > 1e-12
    panel_vertices_np = panel_vertices_np[mask]
    areas_np = areas_np[mask]
    centroids_np = centroids_np[mask]
    N = int(areas_np.shape[0])
    assert N > 1

    # Random surface charge
    sigma_np = rng.standard_normal(size=(N,))

    # Torch tensors on CUDA
    centroids = torch.as_tensor(centroids_np, device=device, dtype=dtype)
    areas = torch.as_tensor(areas_np, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma_np, device=device, dtype=dtype)

    # Build near pairs on CPU
    near_pairs_np = _build_near_pairs_for_panels(
        centroids_np,
        areas_np,
        distance_factor=2.0,
    )
    if near_pairs_np.size == 0:
        # Relax the cutoff in the unlikely case that no pairs are marked "near"
        near_pairs_np = _build_near_pairs_for_panels(
            centroids_np,
            areas_np,
            distance_factor=4.0,
        )
    assert near_pairs_np.shape[1] == 2

    # Base far-field matvec (centroid-lumped) on CUDA, using the same diagonal
    # self-integral correction as the main BEM solver.
    self_integrals = torch.tensor(
        [self_integral_correction(float(a)) for a in areas_np],
        device=device,
        dtype=dtype,
    )

    V_far = bem_matvec_gpu(
        sigma=sigma,
        src_centroids=centroids,
        areas=areas,
        tile_size=64,
        self_integrals=self_integrals,
        use_near_quad=False,
    )

    # CPU near-field correction (reference)
    V_cpu = _apply_near_quadrature_matvec(
        V_far,
        sigma,
        centroids=centroids,
        areas=areas,
        panel_vertices=panel_vertices_np,
        near_pairs=near_pairs_np,
        quad_order=2,
    )

    # CUDA near-field correction
    panel_vertices_cuda = torch.as_tensor(
        panel_vertices_np,
        device=device,
        dtype=dtype,
    )
    near_pairs_cuda = torch.as_tensor(
        near_pairs_np,
        device=device,
        dtype=torch.int64,
    )

    V_cuda = _apply_near_quadrature_matvec_cuda(
        V_far,
        sigma,
        centroids=centroids,
        areas=areas,
        panel_vertices=panel_vertices_cuda,
        near_pairs=near_pairs_cuda,
        quad_order=2,
        panel_vertices_np=panel_vertices_np,
        near_pairs_np=near_pairs_np,
    )

    # Compare
    diff = (V_cpu - V_cuda).norm().item()
    ref = max(1e-12, V_cpu.norm().item())
    rel_err = diff / ref

    # For a purely algebraic rewrite of the same quadrature, we expect agreement
    # to machine precision (up to a bit of floating-point noise).
    assert rel_err < 1e-10
