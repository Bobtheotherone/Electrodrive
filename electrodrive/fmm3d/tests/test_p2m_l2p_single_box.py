from __future__ import annotations

"""
Unit tests for a pure P2M → L2P round-trip in a single box (no tree).

The goal of this file is to provide a *mathematical* sanity check
for the normalized spherical-harmonic basis and multipole conventions
used by the FMM backend, independent of any tree / MAC logic.

We consider the following experiment:

    - A set of source charges q_j at positions x_j inside a ball
      |x_j| <= R_src (center at the origin).

    - A set of target points x_i in a spherical shell
      R_tgt_min <= |x_i| <= R_tgt_max, with R_tgt_min > R_src.

    - The exact potential at x_i is

            φ(x_i) = Σ_j q_j / |x_i - x_j|.

    - We form multipole coefficients M_{lm} up to some order p via

            M_{lm} = Σ_j q_j r_j^l Y_l^m(θ_j, φ_j)^*,

      where (r_j, θ_j, φ_j) are the spherical coordinates of x_j and
      Y_l^m are *normalized* complex spherical harmonics, using the
      same basis as in `electrodrive.fmm3d.spherical_harmonics`.

    - We then evaluate the truncated multipole expansion at the targets:

            φ_mult(x) ≈ Σ_{l=0}^p Σ_{m=-l}^l
                          M_{lm} r^{-(l+1)} Y_l^m(θ, φ).

Under standard multipole theory, the relative L2 error between φ_mult
and φ should decrease roughly like (R_src / R_tgt_min)^(p+1).

These tests check that behavior for p in {4, 6, 8}.
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor

from electrodrive.fmm3d.spherical_harmonics import (
    cartesian_to_spherical,
    spherical_harmonics_complex,
)


# ---------------------------------------------------------------------------
# Random geometry helpers
# ---------------------------------------------------------------------------


def _make_random_charge_cloud(
    n_src: int,
    r_src_max: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int = 1234,
) -> Tuple[Tensor, Tensor]:
    """
    Sample a random cloud of `n_src` charges inside a ball of radius `r_src_max`.

    - Positions are approximately uniform in volume inside the ball.
    - Charges are i.i.d. standard normal.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Sample directions uniformly on the sphere.
    dirs = torch.randn((n_src, 3), generator=g, device=device, dtype=dtype)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # Radii: uniform in volume → r ~ U[0,1]^(1/3) * R_max
    u = torch.rand((n_src,), generator=g, device=device, dtype=dtype)
    radii = r_src_max * u.pow(1.0 / 3.0)

    x_src = radii[:, None] * dirs

    # Charges: zero-mean, unit-variance normal.
    q_src = torch.randn((n_src,), generator=g, device=device, dtype=dtype)

    return x_src, q_src


def _make_random_targets(
    n_tgt: int,
    r_min: float,
    r_max: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int = 4321,
) -> Tensor:
    """
    Sample `n_tgt` random target points in a spherical shell [r_min, r_max].

    The radial distribution is uniform over [r_min, r_max] (not volume-uniform);
    this is sufficient for a robust sanity check.
    """
    assert r_min > 0.0 and r_min < r_max

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Directions: uniform on sphere.
    vec = torch.randn((n_tgt, 3), generator=g, device=device, dtype=dtype)
    vec = vec / vec.norm(dim=-1, keepdim=True)

    # Radii: uniform in [r_min, r_max].
    radii = torch.empty((n_tgt,), device=device, dtype=dtype)
    radii.uniform_(r_min, r_max, generator=g)

    x_tgt = radii[:, None] * vec
    return x_tgt


# ---------------------------------------------------------------------------
# Reference direct potential
# ---------------------------------------------------------------------------


def _direct_potential(
    x_src: Tensor,
    q_src: Tensor,
    x_tgt: Tensor,
) -> Tensor:
    """
    Direct O(N_src * N_tgt) Laplace potential:

        φ(x_i) = Σ_j q_j / |x_i - x_j|.

    No Coulomb constant is applied; we are testing pure 1/r behavior.
    """
    # x_tgt: (N_t, 3), x_src: (N_s, 3)
    diff = x_tgt[:, None, :] - x_src[None, :, :]  # (N_t, N_s, 3)
    r = diff.norm(dim=-1)  # (N_t, N_s)

    # In this setup, targets are strictly outside the source ball, so r>0.
    # Guard against rare numerical zero just in case.
    eps = torch.finfo(r.dtype).eps
    inv_r = 1.0 / torch.clamp(r, min=eps)

    phi = (inv_r * q_src[None, :]).sum(dim=1)  # (N_t,)
    return phi


def _relative_l2_error(phi_ref: Tensor, phi_approx: Tensor) -> float:
    """
    Relative L2 error ||phi_approx - phi_ref|| / ||phi_ref||, as a Python float.
    """
    num = (phi_approx - phi_ref).norm().item()
    den = phi_ref.norm().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


# ---------------------------------------------------------------------------
# Analytic multipole helpers (matching FMM's spherical-harmonic basis)
# ---------------------------------------------------------------------------


def _compute_multipole_coeffs(
    x_src: Tensor,
    q_src: Tensor,
    p: int,
) -> Tensor:
    """
    Compute normalized complex multipole coefficients M_{lm} up to order `p`,
    using the *same* normalized spherical harmonics basis as the FMM backend.

    Convention:
        M_{lm} = Σ_j q_j r_j^l Y_l^m(θ_j, φ_j)^*,

    where (r_j, θ_j, φ_j) are the spherical coordinates of x_j.

    Returns
    -------
    M : Tensor[complex128] with shape (p+1, 2p+1)
        Layout: M[l, m_index], where

            m_index = m + p,  m ∈ [-p, p],

        and entries with |m| > l are explicitly zeroed.
    """
    device = x_src.device
    float_dtype = x_src.dtype  # e.g., torch.float64

    # Spherical coordinates of sources (N_s,)
    r_src, theta_src, phi_src = cartesian_to_spherical(x_src)

    # Y: (N_s, p+1, 2p+1), normalized basis by construction.
    # spherical_harmonics_complex uses the repo's fixed signature:
    #   spherical_harmonics_complex(l_max, theta, phi)
    Y = spherical_harmonics_complex(p, theta_src, phi_src)
    Y = Y.to(torch.complex128)

    q_src_c = q_src.to(torch.complex128)

    # r_j^l for l = 0..p, shape (N_s, p+1)
    N_s = x_src.shape[0]
    l_vals = torch.arange(p + 1, device=device, dtype=float_dtype)  # (p+1,)
    r_pow = r_src[:, None] ** l_vals[None, :]  # (N_s, p+1)

    # Broadcast q_j r_j^l over m: (N_s, p+1, 1)
    weights = (q_src_c[:, None] * r_pow.to(torch.complex128))[:, :, None]

    # M_raw[l, m] = Σ_j q_j r_j^l Y_l^m(θ_j, φ_j)^*
    # Y has layout (N_s, l, m_index).
    M = (weights.conj() * Y.conj()).sum(dim=0)  # (p+1, 2p+1)

    # Explicitly zero out entries with |m| > l to match the FMM packing.
    for l in range(p + 1):
        for m_index in range(2 * p + 1):
            m = m_index - p
            if abs(m) > l:
                M[l, m_index] = 0.0

    return M


def _evaluate_multipole_potential(
    M: Tensor,
    x_tgt: Tensor,
    p: int,
) -> Tensor:
    """
    Evaluate the truncated multipole expansion at target points `x_tgt`.

    We assume:

        - Expansion center is at the origin.
        - M has shape (p+1, 2p+1) and follows the same convention as
          `_compute_multipole_coeffs`.

    Formula:

        φ(x) ≈ Σ_{l=0}^p Σ_{m=-l}^l M_{lm} r^{-(l+1)} Y_l^m(θ, φ),

    with normalized complex spherical harmonics Y_l^m, using the same
    basis as `spherical_harmonics_complex`.
    """
    device = x_tgt.device
    float_dtype = x_tgt.dtype

    # Spherical coords of targets relative to expansion center (origin).
    r_tgt, theta_tgt, phi_tgt = cartesian_to_spherical(x_tgt)

    # Y_tgt: (N_t, p+1, 2p+1)
    Y_tgt = spherical_harmonics_complex(p, theta_tgt, phi_tgt)
    Y_tgt = Y_tgt.to(torch.complex128)

    M_c = M.to(torch.complex128)  # (p+1, 2p+1)

    N_t = x_tgt.shape[0]
    l_vals = torch.arange(p + 1, device=device, dtype=float_dtype)  # (p+1,)

    # r^{-(l+1)} for each target: (N_t, p+1)
    # r_tgt is strictly > 0 here (targets outside source ball).
    r_pow_neg = r_tgt[:, None] ** (-(l_vals + 1.0))[None, :]

    # φ(x) ≈ Σ_{l,m} M_{lm} r^{-(l+1)} Y_l^m(θ, φ).
    # Implement via einsum:
    #   Y_tgt:    (N_t, L, M)
    #   M_c:      (L,   M)
    #   r_pow_neg:(N_t, L)
    phi_complex = torch.einsum(
        "nlm,lm,nl->n",
        Y_tgt,
        M_c,
        r_pow_neg.to(torch.complex128),
    )

    # Potential is real for real charges; discard tiny imaginary drift.
    return phi_complex.real.to(float_dtype)


# ---------------------------------------------------------------------------
# Main test: P2M → L2P round-trip in a single box
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p, tol", [(4, 5e-4), (6, 5e-5), (8, 5e-6)])
def test_single_center_p2m_l2p_roundtrip(p: int, tol: float) -> None:
    """
    Single-center P2M → L2P (mathematical) round-trip:

    - Sources inside |x| <= R_src.
    - Targets with |x| in [R_tgt_min, R_tgt_max], R_tgt_min > R_src.
    - Compare direct 1/r potential with truncated multipole expansion.

    Expected:
        Relative L2 error decreases roughly like (R_src / R_tgt_min)^(p+1),
        so for a fixed geometry, increasing `p` should dramatically reduce
        the error. The specific tolerances below are tuned for float64.
    """
    device = torch.device("cpu")
    dtype = torch.float64

    # Geometry: keep R_src / R_tgt_min small enough that p=4..8 converges fast.
    R_src = 0.4
    R_tgt_min = 1.5
    R_tgt_max = 3.0

    n_src = 64
    n_tgt = 128

    # Random source cloud and targets.
    x_src, q_src = _make_random_charge_cloud(
        n_src=n_src,
        r_src_max=R_src,
        device=device,
        dtype=dtype,
        seed=1234,
    )
    x_tgt = _make_random_targets(
        n_tgt=n_tgt,
        r_min=R_tgt_min,
        r_max=R_tgt_max,
        device=device,
        dtype=dtype,
        seed=4321,
    )

    # Sanity: ensure all targets are outside the source ball.
    r_src_max = x_src.norm(dim=-1).max().item()
    r_tgt_min = x_tgt.norm(dim=-1).min().item()
    assert r_tgt_min > r_src_max, (
        f"Targets must lie outside source ball: r_tgt_min={r_tgt_min:.3e} "
        f"<= r_src_max={r_src_max:.3e}"
    )

    # Exact reference potential.
    phi_ref = _direct_potential(x_src, q_src, x_tgt)

    # P2M: charges -> multipole coefficients around origin.
    M = _compute_multipole_coeffs(x_src, q_src, p=p)

    # L2P: evaluate multipole expansion at targets.
    phi_mult = _evaluate_multipole_potential(M, x_tgt, p=p)

    rel_err = _relative_l2_error(phi_ref, phi_mult)

    # Useful for local debugging if the test is too tight.
    # print(f"[p={p}] rel_l2_err={rel_err:.3e}")

    assert rel_err < tol, (
        f"P2M→L2P round-trip relative L2 error too large for p={p}: "
        f"got {rel_err:.3e}, expected < {tol:.3e}"
    )
