from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Core config / tree / interaction list imports
# ---------------------------------------------------------------------------
from electrodrive.utils.config import K_E  # kept for compatibility; not used in l2p
from electrodrive.fmm3d.config import FmmConfig
from electrodrive.fmm3d.tree import FmmTree
from electrodrive.fmm3d.interaction_lists import build_interaction_lists, InteractionLists

# ---------------------------------------------------------------------------
# Logging imports
# ---------------------------------------------------------------------------
from electrodrive.fmm3d.logging_utils import (
    get_logger,
    ConsoleLogger,
    log_spectral_stats,
    debug_tensor_stats,
    want_verbose_debug,
)

# ---------------------------------------------------------------------------
# Imports from the spherical-harmonics module
# ---------------------------------------------------------------------------
from electrodrive.fmm3d.spherical_harmonics import (
    cartesian_to_spherical,
    spherical_harmonics_complex,
    num_harmonics,
    pack_ylm,
    unpack_ylm,
    index_to_lm,
)

__all__ = [
    "MultipoleOpStats",
    "MultipoleOperators",
    "regular_solid_harmonics_normalized",
    "irregular_solid_harmonics_normalized",
    "MultipoleCoefficients",
    "LocalCoefficients",
    "p2m",
    "m2m",
    "m2l",
    "l2l",
    "l2p",
    "apply_fmm_laplace_potential",
    "p2m_batch",
    "m2l_batch",
    "_make_workspace_for_tree",
]

# Limit on l for which we precompute dense M2L matrices
_M2L_LMAX_LIMIT: int = 32
# Heuristic lower bound to avoid extremely large dimensionless coordinates
_MIN_STABLE_SCALE: float = 1e-1

# ---------------------------------------------------------------------------
# Math Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_l_values_for_packing(p: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Cached array of degrees l for the packed Y_lm ordering up to order p.
    Shape: (P2,), where P2 = (p+1)^2.
    """
    P2 = num_harmonics(p)
    l_values = torch.empty(P2, dtype=dtype, device=device)
    for idx in range(P2):
        l, _ = index_to_lm(idx)
        l_values[idx] = l
    return l_values


def _rescale_multipoles_packed(M_packed: Tensor, ratio: float, p: int) -> Tensor:
    """
    Rescale multipoles M(S_from) -> M(S_to).

    Parameters
    ----------
    M_packed : Tensor, shape (P2,) or (..., P2)
        Packed multipole coefficients at scale S_from.
    ratio : float
        ratio = S_from / S_to.
    p : int
        Expansion order.

    Notes
    -----
    For our P2M convention with dimensionless coordinates y = (x - c)/S:

        R_lm(y) ~ S^{-l} in physical units,
        M_lm(S) ∝ S^{-l}.

    To keep the physical multipole field invariant:

        M_lm(S_to) = (S_from / S_to)^l * M_lm(S_from) = ratio^l * M_lm(S_from).
    """
    if abs(ratio - 1.0) < 1e-9:
        return M_packed

    real_dtype = M_packed.real.dtype if torch.is_complex(M_packed) else M_packed.dtype
    l_values = _get_l_values_for_packing(p, real_dtype, M_packed.device)
    base = torch.as_tensor(ratio, dtype=real_dtype, device=M_packed.device)
    factors = torch.pow(base, l_values)

    if torch.is_complex(M_packed) and not torch.is_complex(factors):
        factors = factors.to(M_packed.dtype)

    return M_packed * factors


def _rescale_locals_packed(L_packed: Tensor, ratio: float, p: int) -> Tensor:
    """
    Rescale locals L(S_from) -> L(S_to).

    Parameters
    ----------
    L_packed : Tensor, shape (P2,) or (..., P2)
        Packed local coefficients at scale S_from.
    ratio : float
        ratio = S_from / S_to.
    p : int
        Expansion order.

    Notes
    -----
    Local expansions use the same basis as multipoles:

        V(x) = (1/S) * Σ_{l,m} L_lm(S) R_lm(y),   y = (x - c)/S.

    Requiring the physical field to be invariant under a change of S gives:

        L_lm(S_to) / S_to^{l+1} = L_lm(S_from) / S_from^{l+1}
        ⇒ L_lm(S_to) = L_lm(S_from) * (S_to / S_from)^{l+1}
                     = ratio^{-(l+1)} * L_lm(S_from)

    when ``ratio = S_from / S_to``.

    The tree-level FMM no longer *relies* on this analytic rescaling, but
    we keep it as a diagnostic utility (e.g. for standalone audits).
    """
    if abs(ratio - 1.0) < 1e-9:
        return L_packed

    real_dtype = L_packed.real.dtype if torch.is_complex(L_packed) else L_packed.dtype
    l_values = _get_l_values_for_packing(p, real_dtype, L_packed.device)
    base = torch.as_tensor(ratio, dtype=real_dtype, device=L_packed.device)
    # NOTE: ratio = S_from / S_to ⇒ L(S_to) = ratio^{-(l+1)} * L(S_from)
    factors = torch.pow(base, -(l_values + 1.0))

    if torch.is_complex(L_packed) and not torch.is_complex(factors):
        factors = factors.to(L_packed.dtype)

    return L_packed * factors


# ---------------------------------------------------------------------------
# Solid Harmonics Helpers
# ---------------------------------------------------------------------------


def regular_solid_harmonics_normalized(l_max: int, r: Tensor, Y: Tensor) -> Tensor:
    """
    Compute regular solid harmonics R_lm = r^l * Y_lm (Schmidt-normalized).

    Supports both:
      - 'unpacked' Y with shape (..., l_max+1, 2*l_max+1)
      - 'packed'   Y with shape (..., (l_max+1)^2)

    Parameters
    ----------
    l_max : int
        Maximum degree l.
    r : Tensor
        Radii (dimensionless), with shape broadcastable to Y's leading
        (batch) dimensions.
    Y : Tensor
        Spherical harmonics Y_lm. Either:
        - shape (..., l_max+1, 2*l_max+1)  [unpacked], or
        - shape (..., (l_max+1)^2)         [packed].

    Returns
    -------
    R : Tensor
        Same shape as Y.
    """
    real_dtype = Y.real.dtype if torch.is_complex(Y) else Y.dtype
    if r.dtype != real_dtype:
        r = r.to(real_dtype)

    P2 = (l_max + 1) ** 2
    L = l_max + 1
    M = 2 * l_max + 1
    shape = Y.shape
    nd = Y.ndim

    # Case 1: 'unpacked' - last two dims are (L, M)
    if nd >= 2 and shape[-2] == L and shape[-1] == M:
        # Flatten all leading dims into one 'batch' dimension
        batch_size = Y.numel() // (L * M)
        Y_flat = Y.reshape(batch_size, L, M)

        # r must match the batch size or be scalar
        if r.numel() == batch_size:
            r_flat = r.reshape(batch_size)
        elif r.numel() == 1:
            r_flat = r.reshape(1).expand(batch_size)
        else:
            raise ValueError(
                f"regular_solid_harmonics_normalized: r.numel()={r.numel()} "
                f"not compatible with Y batch size {batch_size} (unpacked case)"
            )

        R_flat = torch.zeros_like(Y_flat)
        r_pow = torch.ones_like(r_flat)
        for l in range(L):
            R_flat[:, l, :] = r_pow.view(-1, 1) * Y_flat[:, l, :]
            r_pow = r_pow * r_flat

        return R_flat.reshape(shape)

    # Case 2: 'packed' - last dim is P2
    if nd >= 1 and shape[-1] == P2:
        batch_size = Y.numel() // P2
        Y_flat = Y.reshape(batch_size, P2)

        if r.numel() == batch_size:
            r_flat = r.reshape(batch_size)
        elif r.numel() == 1:
            r_flat = r.reshape(1).expand(batch_size)
        else:
            raise ValueError(
                f"regular_solid_harmonics_normalized: r.numel()={r.numel()} "
                f"not compatible with Y batch size {batch_size} (packed case)"
            )

        R_flat = torch.zeros_like(Y_flat)
        r_pow = torch.ones_like(r_flat)
        start = 0
        for l in range(l_max + 1):
            count_l = 2 * l + 1
            end = start + count_l
            R_flat[:, start:end] = r_pow.view(-1, 1) * Y_flat[:, start:end]
            r_pow = r_pow * r_flat
            start = end
            if start >= P2:
                break

        return R_flat.reshape(shape)

    raise ValueError(
        f"regular_solid_harmonics_normalized: unexpected Y.shape={shape}, "
        f"expected '...,(l_max+1, 2*l_max+1)' or '..., (l_max+1)^2'"
    )


def irregular_solid_harmonics_normalized(l_max: int, r: Tensor, Y: Tensor) -> Tensor:
    """
    Compute irregular solid harmonics S_lm = r^{-(l+1)} * Y_lm (Schmidt-normalized).

    Supports both:
      - 'unpacked' Y with shape (..., l_max+1, 2*l_max+1)
      - 'packed'   Y with shape (..., (l_max+1)^2)

    Parameters
    ----------
    l_max : int
        Maximum degree l.
    r : Tensor
        Radii (dimensionless), with shape broadcastable to Y's leading
        (batch) dimensions.
    Y : Tensor
        Spherical harmonics Y_lm. Either:
        - shape (..., l_max+1, 2*l_max+1)  [unpacked], or
        - shape (..., (l_max+1)^2)         [packed].

    Returns
    -------
    S : Tensor
        Same shape as Y.
    """
    real_dtype = Y.real.dtype if torch.is_complex(Y) else Y.dtype
    if r.dtype != real_dtype:
        r = r.to(real_dtype)

    # Avoid r = 0
    r_safe = torch.clamp(r, min=1e-15)
    inv_r = 1.0 / r_safe

    P2 = (l_max + 1) ** 2
    L = l_max + 1
    M = 2 * l_max + 1
    shape = Y.shape
    nd = Y.ndim

    # Case 1: 'unpacked' - last two dims are (L, M)
    if nd >= 2 and shape[-2] == L and shape[-1] == M:
        batch_size = Y.numel() // (L * M)
        Y_flat = Y.reshape(batch_size, L, M)

        if inv_r.numel() == batch_size:
            inv_r_flat = inv_r.reshape(batch_size)
        elif inv_r.numel() == 1:
            inv_r_flat = inv_r.reshape(1).expand(batch_size)
        else:
            raise ValueError(
                f"irregular_solid_harmonics_normalized: r.numel()={inv_r.numel()} "
                f"not compatible with Y batch size {batch_size} (unpacked case)"
            )

        S_flat = torch.zeros_like(Y_flat)
        r_pow = inv_r_flat.clone()
        for l in range(L):
            S_flat[:, l, :] = r_pow.view(-1, 1) * Y_flat[:, l, :]
            r_pow = r_pow * inv_r_flat

        return S_flat.reshape(shape)

    # Case 2: 'packed' - last dim is P2
    if nd >= 1 and shape[-1] == P2:
        batch_size = Y.numel() // P2
        Y_flat = Y.reshape(batch_size, P2)

        if inv_r.numel() == batch_size:
            inv_r_flat = inv_r.reshape(batch_size)
        elif inv_r.numel() == 1:
            inv_r_flat = inv_r.reshape(1).expand(batch_size)
        else:
            raise ValueError(
                f"irregular_solid_harmonics_normalized: r.numel()={inv_r.numel()} "
                f"not compatible with Y batch size {batch_size} (packed case)"
            )

        S_flat = torch.zeros_like(Y_flat)
        r_pow = inv_r_flat.clone()
        start = 0
        for l in range(l_max + 1):
            count_l = 2 * l + 1
            end = start + count_l
            S_flat[:, start:end] = r_pow.view(-1, 1) * Y_flat[:, start:end]
            r_pow = r_pow * inv_r_flat
            start = end
            if start >= P2:
                break

        return S_flat.reshape(shape)

    raise ValueError(
        f"irregular_solid_harmonics_normalized: unexpected Y.shape={shape}, "
        f"expected '...,(l_max+1, 2*l_max+1)' or '..., (l_max+1)^2'"
    )


# ---------------------------------------------------------------------------
# Proxy Surface & Translation Matrices
# ---------------------------------------------------------------------------


def _build_proxy_surface(n_theta: int = 16, n_phi: int = 32, radius: float = 1.5) -> Tensor:
    """
    Build a quasi-uniform proxy surface on a sphere of given radius.

    Returns
    -------
    xyz : Tensor, shape (N, 3)
        Cartesian coordinates of proxy points.
    """
    # Older PyTorch versions may not support the `endpoint` keyword in linspace,
    # so we avoid it and emulate [0, 2π) manually.
    thetas = torch.linspace(0.0, math.pi, n_theta)

    # We want n_phi points in [0, 2π) (excluding 2π itself).
    # Use n_phi + 1 samples on [0, 2π] and drop the last one.
    phis_full = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1)
    phis = phis_full[:-1]

    theta_grid, phi_grid = torch.meshgrid(thetas, phis, indexing="ij")

    x = radius * torch.sin(theta_grid) * torch.cos(phi_grid)
    y = radius * torch.sin(theta_grid) * torch.sin(phi_grid)
    z = radius * torch.cos(theta_grid)

    xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    return xyz


def _build_charge_to_multipole_matrix(
    p: int,
    proxy_points: Tensor,
    center: Tensor,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build matrix mapping proxy charges -> packed multipoles.

    Parameters
    ----------
    p : int
        Expansion order.
    proxy_points : Tensor, shape (N_p, 3)
        Cartesian coordinates of proxy points in *physical* (here:
        dimensionless) units.
    center : Tensor, shape (3,)
        Expansion center in the same units.
    scale : float
        Expansion radius S. Coordinates are scaled as y = (x - c)/S
        inside :func:`p2m`.
    device, dtype : torch.device, torch.dtype
        Real-valued device / dtype controlling where computations live.

    Returns
    -------
    P_M : Tensor, shape (P2, N_p)
        Matrix such that, for a column vector of proxy charges ``q``,
        the packed multipoles are

            M_packed = P_M @ q

        using exactly the same convention as :meth:`MultipoleOperators.p2m`.
    """
    P2 = num_harmonics(p)
    pts = proxy_points.to(device=device, dtype=dtype)
    center = center.to(device=device, dtype=dtype)

    # Dimensionless coordinates y = (x - c) / S
    inv_scale = 1.0 / float(scale)
    rel = (pts - center.unsqueeze(0)) * inv_scale  # (N_p, 3)

    r, theta, phi = cartesian_to_spherical(rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    R = regular_solid_harmonics_normalized(p, r, Y)
    R_packed = pack_ylm(R, p)  # (N_p, P2)

    # p2m forms M_lm = sum_i q_i R_lm(y_i), so the linear map is
    # exactly R_packed^T. The multipole convention used throughout the
    # tests (see _p2m_spec) is M_lm = sum q_i R_lm^*(y_i), i.e. the
    # complex-conjugated solid harmonics. We apply the conjugate once
    # here so that the same convention is shared by translation matrices.
    return torch.conj(R_packed).T  # (P2, N_p)


def _build_charge_to_local_matrix(
    p: int,
    proxy_points: Tensor,
    center: Tensor,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    radii: Optional[List[float]] = None,
) -> Tensor:
    """
    Build matrix mapping proxy charges -> packed local coefficients.

    This routine defines the local coefficients *numerically* by
    matching the (dimensionless) Laplace potential generated by
    proxy charges on a small sphere of “check” points inside the
    target box.

    Conventions
    -----------
    - Coordinates are expressed in the same (possibly dimensionless)
      units as everywhere else in this module.
    - The scale ``S`` only enters through the dimensionless
      representation

          y = (x - center) / S,

      but because the Laplace kernel is homogeneous of degree -1, the
      dimensionless potential

          U(y) = S * V(x)

      is simply

          U(y) = sum_j q_j / |y - y_j|,

      where ``y_j`` are the dimensionless source coordinates.
      Consequently, we can work entirely with dimensionless coordinates
      here (and set ``scale = 1`` conceptually) without loss of
      generality.

    Parameters
    ----------
    p : int
        Expansion order.
    proxy_points : Tensor, shape (N_p, 3)
        Source locations representing the far-field charge distribution
        (in the same coordinate system as ``center``).
    center : Tensor, shape (3,)
        Center of the local expansion.
    scale : float
        Expansion radius S (kept for signature compatibility; it does
        not affect the numerics here).
    device, dtype : torch.device, torch.dtype
        Real-valued device / dtype.

    Returns
    -------
    P_L : Tensor, shape (P2, N_p)
        Matrix such that, for a column vector of proxy charges ``q``,
        the packed local coefficients are

            L_packed = P_L @ q

        in the convention used by :meth:`MultipoleOperators.l2p`.
    """
    P2 = num_harmonics(p)

    proxy_points = proxy_points.to(device=device, dtype=dtype)
    center = center.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 1. Choose check points on multiple radii inside the target box.
    #    Using multiple shells prevents high-l numerical instability.
    # ------------------------------------------------------------------
    if radii is None:
        radii = [0.2, 0.4, 0.6, 0.8]

    R_blocks = []
    G_blocks = []

    for r_inner in radii:
        check_local = _build_proxy_surface(radius=r_inner).to(
            device=device, dtype=dtype
        )  # (N_chk, 3)

        # Global coordinates of the check points (still in the same units as
        # proxy_points and center).
        check_global = center.unsqueeze(0) + check_local  # (N_chk, 3)

        # ------------------------------------------------------------------
        # 2. Direct (dimensionless) Laplace potential at check points due
        #    to unit charges at proxy_points:
        #
        #       U(y_k) = sum_j q_j / |x_k - x_j|
        # ------------------------------------------------------------------
        diff = check_global.unsqueeze(1) - proxy_points.unsqueeze(0)  # (N_chk, N_p, 3)
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-15)       # (N_chk, N_p)
        G = 1.0 / dist                                                # (N_chk, N_p)

        # ------------------------------------------------------------------
        # 3. Local basis evaluated at the same check points, in coordinates
        #    relative to the local center.
        #
        #    U(y) = sum_{l,m} L_lm R_lm(y), with y = check_local
        # ------------------------------------------------------------------
        R_check = _local_potential_matrix(p, check_local, device, dtype)  # (N_chk, P2)

        R_blocks.append(R_check)
        G_blocks.append(G)

    # Concatenate all blocks to form a robust overdetermined system
    R_big = torch.cat(R_blocks, dim=0)  # (N_tot, P2)
    G_big = torch.cat(G_blocks, dim=0)  # (N_tot, N_p)

    # Match dtypes: solve in complex arithmetic with G promoted to the
    # same (complex) dtype as R_big.
    G_c = G_big.to(R_big.dtype)

    # Solve R_big @ P_L ≈ G_c in the least-squares sense to obtain a
    # single linear map from charges to local coefficients.
    P_L = torch.linalg.lstsq(R_big, G_c).solution  # (P2, N_p)
    assert P_L.shape == (P2, proxy_points.shape[0])
    return P_L


def _multipole_potential_matrix(
    p: int,
    coords: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build a dense matrix mapping multipole coefficients -> potentials
    at a set of evaluation points.

    Parameters
    ----------
    p : int
        Expansion order.
    coords : Tensor, shape (N, 3)
        Evaluation points in *dimensionless* coordinates relative to
        the multipole center.
    device, dtype : torch.device, torch.dtype
        Real-valued device / dtype controlling where computations live.

    Returns
    -------
    A : Tensor, shape (N, P2)
        Each row contains the irregular solid harmonics S_lm(y_k)
        evaluated at coords[k].
    """
    coords = coords.to(device=device, dtype=dtype)
    if coords.numel() == 0:
        P2 = num_harmonics(p)
        return torch.zeros(
            0,
            P2,
            dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            device=device,
        )

    r, theta, phi = cartesian_to_spherical(coords)
    Y = spherical_harmonics_complex(p, theta, phi)
    S = irregular_solid_harmonics_normalized(p, r, Y)
    S_packed = pack_ylm(S, p)
    return S_packed


def _local_potential_matrix(
    p: int,
    coords: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build a dense matrix mapping local coefficients -> potentials at a
    set of evaluation points.

    Parameters
    ----------
    p : int
        Expansion order.
    coords : Tensor, shape (N, 3)
        Evaluation points in *dimensionless* coordinates relative to
        the local expansion center.
    device, dtype : torch.device, torch.dtype
        Real-valued device / dtype controlling where computations live.

    Returns
    -------
    B : Tensor, shape (N, P2)
        Each row contains the regular solid harmonics R_lm(y_k)
        evaluated at coords[k].
    """
    coords = coords.to(device=device, dtype=dtype)
    if coords.numel() == 0:
        P2 = num_harmonics(p)
        return torch.zeros(
            0,
            P2,
            dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            device=device,
        )

    r, theta, phi = cartesian_to_spherical(coords)
    Y = spherical_harmonics_complex(p, theta, phi)
    R = regular_solid_harmonics_normalized(p, r, Y)
    R_packed = pack_ylm(R, p)
    return R_packed


def _compute_m2m_matrix(
    p: int,
    t_vec: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Compute dense M2M translation matrix using proxy charges.

    Instead of collocating potentials from abstract multipole
    coefficients, we enforce that, for a set of proxy charges placed
    on a sphere enclosing the source box, the multipole expansion
    about the child center and the multipole expansion about the
    parent center represent the *same* exterior field.

    Concretely, let ``q`` be charges located at proxy points
    ``proxy_points``.  Denote

        M_child  = P_child @ q
        M_parent = P_parent @ q

    the packed multipoles about the child and parent centers,
    respectively, where ``P_child`` and ``P_parent`` are built using
    the same P2M convention as :meth:`MultipoleOperators.p2m`.

    We then determine ``T`` from

        M_parent = T @ M_child     for all q

    which in matrix form is

        P_parent = T @ P_child.

    This determines ``T`` (in the least-squares sense when the system
    is overdetermined), and ensures that M2M is *exactly* consistent
    with P2M under the chosen truncation.

    Parameters
    ----------
    p : int
        Expansion order.
    t_vec : Tensor, shape (3,)
        Dimensionless translation vector (parent_center - child_center)
        in the common length scale used for the expansions.
    device, dtype : torch.device, torch.dtype

    Returns
    -------
    T : Tensor, shape (P2, P2)
        Packed M2M translation matrix such that

            M_parent = T @ M_child.
    """
    if p > _M2L_LMAX_LIMIT:
        raise ValueError(f"M2M matrix for p={p} is not cached (limit={_M2L_LMAX_LIMIT}).")

    P2 = num_harmonics(p)

    # Strip any imaginary part (should be zero) before casting to real dtype.
    t_vec = t_vec.real.to(device=device, dtype=dtype)

    # Child center at the origin, parent at t_vec.  We place proxy
    # charges on a sphere strictly inside the unit ball, so that the
    # multipole expansion converges everywhere outside that sphere.
    r_src = 0.5
    proxy_points = _build_proxy_surface(radius=r_src).to(device=device, dtype=dtype)  # (N_p, 3)

    center_child = torch.zeros(3, device=device, dtype=dtype)
    center_parent = t_vec

    # Charge-to-multipole matrices for child and parent centers.
    P_child = _build_charge_to_multipole_matrix(
        p, proxy_points, center_child, scale=1.0, device=device, dtype=dtype
    )  # (P2, N_p)
    P_parent = _build_charge_to_multipole_matrix(
        p, proxy_points, center_parent, scale=1.0, device=device, dtype=dtype
    )  # (P2, N_p)

    # Solve P_parent ≈ T @ P_child  =>  P_parent^T ≈ P_child^T @ T^T.
    A = P_child.T  # (N_p, P2)
    B = P_parent.T  # (N_p, P2)

    # Row-normalize for improved conditioning.
    row_norm = torch.linalg.norm(A, dim=1).clamp(min=1e-15).unsqueeze(1)
    A_scaled = A / row_norm
    B_scaled = B / row_norm

    T_T = torch.linalg.lstsq(A_scaled, B_scaled).solution  # (P2, P2)
    T = T_T.T
    assert T.shape == (P2, P2)
    return T


def _compute_m2l_matrix(
    p: int,
    t_vec: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Proxy-based M2L translation via equivalent charges.

    This version avoids directly collocating irregular solid harmonics
    S_lm(t + y_k) on interior check points, which can become severely
    ill-conditioned when |t + y_k| is small. Instead we:

      1. Place a set of proxy charges q_j on a sphere enclosing the
         source box.
      2. Build their multipole coefficients about the source center via
         the same P2M convention as :meth:`MultipoleOperators.p2m`,
         giving

             M = P_M @ q

         for a (P2, N_p) matrix P_M.
      3. Build their local coefficients about the target center by
         matching direct 1/|r| potentials on interior check points via
         :func:`_build_charge_to_local_matrix`, giving

             L = P_L @ q

         for a (P2, N_p) matrix P_L.
      4. Determine the M2L matrix T_m2l from

             L ≈ T_m2l @ M   for all q,

         i.e. in matrix form

             P_L ≈ T_m2l @ P_M  ⇒  P_L^T ≈ P_M^T @ T_m2l^T.

    Solving this overdetermined linear system in the least-squares sense
    makes M2L exactly consistent (under truncation) with the P2M and
    charge→local maps used elsewhere, and is robust even when the source
    and target boxes are not extremely well separated in the chosen
    dimensionless scaling.

    Parameters
    ----------
    p : int
        Expansion order.
    t_vec : Tensor, shape (3,)
        Dimensionless translation (target_center - source_center) / S.
    device, dtype : torch.device, torch.dtype
        Real-valued device / dtype controlling where computations live.

    Returns
    -------
    T_m2l : Tensor, shape (P2, P2)
        Packed M2L translation matrix such that

            L_packed = T_m2l @ M_packed.
    """
    if p > _M2L_LMAX_LIMIT:
        raise ValueError(
            f"M2L matrix for p={p} is not cached (limit={_M2L_LMAX_LIMIT})."
        )

    # Strip any tiny imaginary component and cast to the requested real dtype.
    t_vec = t_vec.real.to(device=device, dtype=dtype)
    norm_t = float(torch.linalg.norm(t_vec))

    # Degenerate translations should never occur in a well-formed M2L list,
    # but guard against them defensively.
    if norm_t < 1e-12:
        raise ValueError("M2L translation requires non-zero separation vector.")

    P2 = num_harmonics(p)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    # 1. Equivalent sources on a proxy surface around the source box.
    #
    #    Place the proxy sphere inside the unit ball (scale=1.0) and adapt
    #    its radius to the separation. Using a fixed 0.5 radius for all
    #    pairs overestimates the source extent for small boxes, which in
    #    turn forces the check radii outside the convergence region when
    #    |t| is small. Shrinking r_proxy toward |t|/2 keeps the geometry
    #    well separated without sacrificing far-field accuracy.
    base_proxy = 0.5
    r_proxy = min(base_proxy, 0.5 * norm_t)
    proxy_points = _build_proxy_surface(radius=r_proxy).to(device=device, dtype=dtype)

    # Adapt the check-sphere radii so that they remain inside the
    # convergence region |y| < |t| - r_proxy for near-threshold pairs.
    scale_proxy = r_proxy / base_proxy if base_proxy > 0 else 1.0
    base_radii = [0.2 * scale_proxy, 0.4 * scale_proxy, 0.6 * scale_proxy, 0.8 * scale_proxy]
    r_conv = max(norm_t - r_proxy, 1e-3)
    scale_r = min(1.0, r_conv / max(base_radii))
    radii = [max(r * scale_r, 1e-3) for r in base_radii]

    center_src = torch.zeros(3, device=device, dtype=dtype)
    center_tgt = t_vec

    # Charge-to-multipole for the source center.
    P_M = _build_charge_to_multipole_matrix(
        p,
        proxy_points,
        center_src,
        scale=1.0,
        device=device,
        dtype=dtype,
    )  # (P2, N_p)

    # Charge-to-local for the target center.
    P_L = _build_charge_to_local_matrix(
        p,
        proxy_points,
        center_tgt,
        scale=1.0,
        device=device,
        dtype=dtype,
        radii=radii,
    )  # (P2, N_p)

    # 2. Solve P_L ≈ T_m2l @ P_M  in least squares.
    #
    #    Writing A = P_M^T, B = P_L^T, we solve
    #
    #        A @ T_m2l^T ≈ B
    #
    #    row-wise normalized for improved conditioning.
    A = P_M.T  # (N_p, P2)
    B = P_L.T  # (N_p, P2)

    row_norm = torch.linalg.norm(A, dim=1).clamp(min=1e-15).unsqueeze(1)
    A_scaled = A / row_norm
    B_scaled = B / row_norm

    T_T = torch.linalg.lstsq(A_scaled, B_scaled).solution  # (P2, P2)
    T_m2l = T_T.T.to(complex_dtype)
    assert T_m2l.shape == (P2, P2)
    return T_m2l


def _compute_l2l_matrix(
    p: int,
    t_vec: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Compute dense L2L translation matrix by direct collocation of local
    expansions.

    We enforce that, for a set of check points shared between the parent
    and child boxes, the local expansion about the parent center and the
    local expansion about the child center represent the *same* analytic
    field.

    Concretely, letting L_parent be the local coefficients at the parent
    center and L_child those at the child center, we require

        U(x_k) = sum_{l,m} L_parent_{lm} R_{lm}(y_p,k)
               = sum_{l,m} L_child_{lm}  R_{lm}(y_c,k)

    for all check points x_k, where

        y_c,k = x_k - c_child,
        y_p,k = x_k - c_parent = y_c,k + t_vec,

    and t_vec = c_child - c_parent (dimensionless).

    In matrix form, with

        B_parent[k,:] = R_{lm}(y_p,k),
        B_child[k,:]  = R_{lm}(y_c,k),

    we seek T_l2l such that

        B_child @ T_l2l ≈ B_parent

    in the least-squares sense, so that

        L_child = T_l2l @ L_parent
        ⇒ U_child(x_k) = U_parent(x_k) for all k.
    """
    if p > _M2L_LMAX_LIMIT:
        raise ValueError(
            f"L2L matrix for p={p} is not cached (limit={_M2L_LMAX_LIMIT})."
        )

    # Strip any tiny imaginary part and cast to the requested real dtype.
    t_vec = t_vec.real.to(device=device, dtype=dtype)

    P2 = num_harmonics(p)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    # Degenerate translation: child == parent ⇒ identity.
    norm_t = float(torch.linalg.norm(t_vec))
    if norm_t < 1e-12:
        return torch.eye(P2, dtype=complex_dtype, device=device)

    # 1. Build check points relative to the *child* center.
    #    We reuse the proxy-surface helper and place several shells
    #    strictly inside the unit ball.
    radii = [0.25, 0.5, 0.8]
    coords_child_list: List[Tensor] = []
    for r_inner in radii:
        shell = _build_proxy_surface(radius=r_inner).to(device=device, dtype=dtype)
        coords_child_list.append(shell)
    coords_child = torch.cat(coords_child_list, dim=0)  # (N_chk, 3)

    # 2. The same points in coordinates relative to the *parent* center.
    coords_parent = coords_child + t_vec.unsqueeze(0)  # (N_chk, 3)

    # 3. Build local potential matrices for parent and child.
    B_child = _local_potential_matrix(p, coords_child, device, dtype)   # (N_chk, P2)
    B_parent = _local_potential_matrix(p, coords_parent, device, dtype) # (N_chk, P2)

    # 4. Solve B_child @ T ≈ B_parent in least squares, with row scaling
    #    for better conditioning (similar to M2L construction).
    row_norm = torch.linalg.norm(B_child, dim=1).clamp(min=1e-15).unsqueeze(1)
    B_child_scaled = B_child / row_norm
    B_parent_scaled = B_parent / row_norm

    T_l2l = torch.linalg.lstsq(B_child_scaled, B_parent_scaled).solution  # (P2, P2)
    assert T_l2l.shape == (P2, P2)
    return T_l2l


@dataclass
class MultipoleOpStats:
    """
    Lightweight counters to track multipole operator usage.

    This class is intentionally simple but provides the interface expected
    by other FMM modules (e.g. kernels_cpu), namely:

      - incr(name: str, value: float)
      - merge_from(other: MultipoleOpStats)
      - to_dict()

    The fields are stored as floats so we can accumulate fractional
    contributions if desired.
    """
    def __init__(self) -> None:
        self.p2m_calls: float = 0.0
        self.m2m_calls: float = 0.0
        self.m2l_calls: float = 0.0
        self.l2l_calls: float = 0.0
        self.l2p_calls: float = 0.0
        # Optional extras for telemetry
        self.extras: Dict[str, float] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        """
        Increment the given counter by `value`. The name is expected to
        match one of the known fields (e.g. "p2m_calls").
        """
        if not hasattr(self, name):
            raise AttributeError(f"Unknown MultipoleOpStats field {name!r}")
        current = getattr(self, name)
        setattr(self, name, current + value)

    def merge_from(self, other: "MultipoleOpStats") -> None:
        """
        Accumulate statistics from another MultipoleOpStats instance.
        """
        if other is None:
            return
        self.p2m_calls += other.p2m_calls
        self.m2m_calls += other.m2m_calls
        self.m2l_calls += other.m2l_calls
        self.l2l_calls += other.l2l_calls
        self.l2p_calls += other.l2p_calls
        for k, v in getattr(other, "extras", {}).items():
            self.extras[k] = self.extras.get(k, 0.0) + float(v)

    def to_dict(self) -> Dict[str, float]:
        out = {
            "p2m_calls": float(self.p2m_calls),
            "m2m_calls": float(self.m2m_calls),
            "m2l_calls": float(self.m2l_calls),
            "l2l_calls": float(self.l2l_calls),
            "l2p_calls": float(self.l2p_calls),
        }
        out.update(self.extras)
        return out


def _accumulate_stats(dst: Optional[MultipoleOpStats], src: MultipoleOpStats) -> None:
    if dst is None or src is None:
        return
    dst.merge_from(src)


# ---------------------------------------------------------------------------
# Core Multipole Operator
# ---------------------------------------------------------------------------


class MultipoleOperators:
    """
    Core Laplace FMM operators in 3D using Schmidt-normalized Y_lm.

    This class implements P2M, M2M, M2L, L2L, and L2P for the 1/|r| kernel.
    Any physical constants (e.g. K_E) are applied *outside* this class.
    """

    def __init__(
        self,
        p: int,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        logger: Optional[ConsoleLogger] = None,
    ):
        self.p = int(p)
        self.P2 = num_harmonics(self.p)
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.device = device if device is not None else torch.device("cpu")

        self.logger = logger if logger is not None else get_logger()
        self.stats = MultipoleOpStats()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _prepare_inputs(self, *xs: Tensor, ensure_complex: bool = False) -> Tuple[Tensor, ...]:
        out: List[Tensor] = []
        for x in xs:
            if not isinstance(x, Tensor):
                x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
            else:
                x = x.to(device=self.device)
                if x.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
                    x = x.to(self.dtype)

            if ensure_complex and not torch.is_complex(x):
                x = x.to(self.complex_dtype)
            out.append(x)
        return tuple(out)

    # -----------------------------------------------------------------------
    # P2M
    # -----------------------------------------------------------------------

    def p2m(
        self,
        src: Tensor,
        q: Tensor,
        center: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Particle-to-multipole (P2M) expansion for the 1/|r| kernel.

        Parameters
        ----------
        src : Tensor, shape (N, 3)
            Source coordinates in physical units.
        q : Tensor, shape (N,)
            Source strengths.
        center : Tensor, shape (3,)
            Expansion center in physical units.
        scale : float
            Expansion radius S. Coordinates are scaled as y = (x - c)/S.

        Returns
        -------
        M_packed : Tensor, shape (P2,)
            Packed multipole coefficients.
        """
        self.stats.p2m_calls += 1
        p = self.p

        if scale <= 1e-9 or src.shape[0] == 0:
            return torch.zeros(self.P2, dtype=self.complex_dtype, device=self.device)

        src, q, center = self._prepare_inputs(src, q, center)

        # --- Ensure center is a flat 3-vector (handles e.g. (1,3) input) ---
        center = center.view(-1)
        if center.numel() != 3:
            raise ValueError(
                f"P2M expects 'center' to have 3 elements, got shape {tuple(center.shape)}"
            )

        inv_scale = 1.0 / float(scale)
        rel = (src - center.unsqueeze(0)) * inv_scale
        r, theta, phi = cartesian_to_spherical(rel)
        Y = spherical_harmonics_complex(p, theta, phi)
        R = regular_solid_harmonics_normalized(p, r, Y)
        R_packed = pack_ylm(R, p)

        q_c = q.to(self.complex_dtype)
        # Multipole convention matches the spec helper in tests:
        # M_lm = sum_i q_i * R_lm^*(y_i) with Schmidt-normalized Y_lm.
        M_packed = torch.sum(q_c.unsqueeze(-1) * torch.conj(R_packed), dim=0)

        if want_verbose_debug():
            debug_tensor_stats("P2M_M", M_packed, self.logger)
            log_spectral_stats(self.logger, "P2M", M_packed.unsqueeze(0), p)

        return M_packed

    # -----------------------------------------------------------------------
    # M2M
    # -----------------------------------------------------------------------

    def m2m(
        self,
        M_child: Tensor,
        t_vec: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Multipole-to-multipole (M2M) translation.

        Parameters
        ----------
        M_child : Tensor, shape (P2,)
            Child multipole coefficients about its own center.
        t_vec : Tensor, shape (3,)
            Physical translation (parent_center - child_center).
        scale : float
            Expansion radius S for the shared global expansion basis. The
            translation is internally normalized by ``1/scale``.

        Returns
        -------
        M_parent : Tensor, shape (P2,)
            Parent multipole coefficients about the parent center.
        """
        self.stats.m2m_calls += 1
        (M_child, t_vec) = self._prepare_inputs(M_child, t_vec, ensure_complex=True)
        p = self.p

        if torch.allclose(M_child, torch.zeros_like(M_child)):
            return M_child.clone()

        scale_work = max(float(scale), _MIN_STABLE_SCALE)
        need_rescale = abs(scale_work - float(scale)) > 1e-14
        M_in = (
            _rescale_multipoles_packed(M_child, float(scale) / scale_work, p)
            if need_rescale
            else M_child
        )

        t_dimless = t_vec.real * (1.0 / scale_work)
        T = _compute_m2m_matrix(p, t_dimless, self.device, self.dtype)
        M_parent = T @ M_in

        if need_rescale:
            M_parent = _rescale_multipoles_packed(M_parent, scale_work / float(scale), p)

        if want_verbose_debug():
            debug_tensor_stats("M2M_child", M_child, self.logger)
            debug_tensor_stats("M2M_parent", M_parent, self.logger)
            log_spectral_stats(self.logger, "M2M", M_parent.unsqueeze(0), p)

        return M_parent

    # -----------------------------------------------------------------------
    # M2L
    # -----------------------------------------------------------------------

    def m2l(
        self,
        M_source: Tensor,
        t_vec: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Multipole-to-local (M2L) translation for the 1/|r| kernel.

        Parameters
        ----------
        M_source : Tensor, shape (P2,)
            Source multipole coefficients about source center.
        t_vec : Tensor, shape (3,)
            Physical translation (target_center - source_center).
        scale : float
            Expansion radius S shared by multipole and local bases. The
            translation is internally normalized by ``1/scale``.

        Returns
        -------
        L_target : Tensor, shape (P2,)
            Local expansion coefficients at the target center.
        """
        self.stats.m2l_calls += 1
        (M_source, t_vec) = self._prepare_inputs(M_source, t_vec, ensure_complex=True)
        p = self.p

        if torch.allclose(M_source, torch.zeros_like(M_source)):
            return torch.zeros_like(M_source)

        scale_work = max(float(scale), _MIN_STABLE_SCALE)
        need_rescale = abs(scale_work - float(scale)) > 1e-14
        M_in = (
            _rescale_multipoles_packed(M_source, float(scale) / scale_work, p)
            if need_rescale
            else M_source
        )

        t_dimless = t_vec.real * (1.0 / scale_work)
        T_m2l = _compute_m2l_matrix(p, t_dimless, self.device, self.dtype)
        L_target = T_m2l @ M_in

        if need_rescale:
            L_target = _rescale_locals_packed(L_target, scale_work / float(scale), p)

        if want_verbose_debug():
            debug_tensor_stats("M2L_M_source", M_source, self.logger)
            debug_tensor_stats("M2L_L_target", L_target, self.logger)
            log_spectral_stats(self.logger, "M2L", L_target.unsqueeze(0), p)

        return L_target

    # -----------------------------------------------------------------------
    # L2L
    # -----------------------------------------------------------------------

    def l2l(
        self,
        L_parent: Tensor,
        t_vec: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Local-to-local (L2L) translation.

        Parameters
        ----------
        L_parent : Tensor, shape (P2,)
            Local expansion at parent center.
        t_vec : Tensor, shape (3,)
            Physical translation (child_center - parent_center).
        scale : float
            Expansion radius S (shared by parent and child locals). The
            translation is internally normalized by ``1/scale``.

        Returns
        -------
        L_child : Tensor, shape (P2,)
            Local expansion at child center.
        """
        self.stats.l2l_calls += 1
        (L_parent, t_vec) = self._prepare_inputs(L_parent, t_vec, ensure_complex=True)
        p = self.p

        if torch.allclose(L_parent, torch.zeros_like(L_parent)):
            return torch.zeros_like(L_parent)

        scale_work = max(float(scale), _MIN_STABLE_SCALE)
        need_rescale = abs(scale_work - float(scale)) > 1e-14
        L_in = (
            _rescale_locals_packed(L_parent, float(scale) / scale_work, p)
            if need_rescale
            else L_parent
        )

        t_dimless = t_vec.real * (1.0 / scale_work)
        T_l2l = _compute_l2l_matrix(p, t_dimless, self.device, self.dtype)
        L_child = T_l2l @ L_in

        if need_rescale:
            L_child = _rescale_locals_packed(L_child, scale_work / float(scale), p)

        if want_verbose_debug():
            debug_tensor_stats("L2L_parent", L_parent, self.logger)
            debug_tensor_stats("L2L_child", L_child, self.logger)
            log_spectral_stats(self.logger, "L2L", L_child.unsqueeze(0), p)

        return L_child

    # -----------------------------------------------------------------------
    # L2P
    # -----------------------------------------------------------------------

    def l2p(
        self,
        L_packed: Tensor,
        targets: Tensor,
        center: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Local-to-potential evaluation (L2P) for the 1/|r| kernel.

        Parameters
        ----------
        L_packed : Tensor, shape (P2,)
            Local coefficients at 'center' with radius 'scale'.
        targets : Tensor, shape (N, 3)
            Target coordinates in physical units.
        center : Tensor, shape (3,)
            Local expansion center in physical units.
        scale : float
            Local/global expansion radius S.

        Returns
        -------
        V : Tensor, shape (N,)
            Real-valued potentials (same dtype as operator), representing
            the pure 1/|r| kernel. Any Coulomb constant K_E is applied
            outside this function (e.g. in BEM/FMM glue code).
        """
        self.stats.l2p_calls += 1
        p = self.p

        if scale <= 1e-9 or L_packed.norm() == 0:
            return torch.zeros(targets.shape[0], dtype=self.dtype, device=self.device)

        (L_packed,) = self._prepare_inputs(L_packed, ensure_complex=True)
        (targets, center) = self._prepare_inputs(targets, center)

        # --- Ensure center is a flat 3-vector (handles e.g. (1,3) input) ---
        center = center.view(-1)
        if center.numel() != 3:
            raise ValueError(
                f"L2P expects 'center' to have 3 elements, got shape {tuple(center.shape)}"
            )

        inv_scale = 1.0 / float(scale)
        rel_coords = (targets - center.unsqueeze(0)) * inv_scale
        r, theta, phi = cartesian_to_spherical(rel_coords)
        Y = spherical_harmonics_complex(p, theta, phi)
        R_schmidt = regular_solid_harmonics_normalized(p, r, Y)
        R_packed = pack_ylm(R_schmidt, p)

        # Dimensionless local potential: U(y) = Σ L_lm R_lm(y)
        U_complex = torch.sum(L_packed.unsqueeze(0) * R_packed, dim=-1)

        # Convert to physical potential for the 1/|r| kernel: V(x) = U(y) / S
        V = U_complex.real.to(self.dtype) * inv_scale

        if want_verbose_debug():
            debug_tensor_stats("L2P_L", L_packed, self.logger)
            debug_tensor_stats("L2P_V", V, self.logger)

        return V

    # -----------------------------------------------------------------------
    # Batch stubs (kept for API compatibility)
    # -----------------------------------------------------------------------

    def p2m_batch(self, *args, **kwargs):
        raise NotImplementedError("p2m_batch not implemented")

    def m2l_batch(self, *args, **kwargs):
        raise NotImplementedError("m2l_batch not implemented")


# ---------------------------------------------------------------------------
# Tree-aware Data Structures & Wrappers
# ---------------------------------------------------------------------------


@dataclass
class MultipoleCoefficients:
    data: Tensor
    tree: FmmTree
    charges: Tensor  # in tree order
    p: int
    dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass
class LocalCoefficients:
    data: Tensor
    tree: FmmTree
    p: int
    dtype: torch.dtype
    complex_dtype: torch.dtype


def _validate_fmm_config(cfg: FmmConfig) -> None:
    if cfg.expansion_order < 0:
        raise ValueError("FMM expansion_order must be non-negative.")
    if cfg.mac_theta <= 0.0 or cfg.mac_theta >= 1.0:
        raise ValueError("FMM mac_theta must be in (0, 1).")


def _make_workspace_for_tree(tree: FmmTree, cfg: FmmConfig) -> MultipoleOperators:
    _validate_fmm_config(cfg)
    cfg_dtype = getattr(cfg, "dtype", None)
    base_dtype = (
        cfg_dtype
        if cfg_dtype is not None
        else (tree.points.dtype if tree.points.dtype in (torch.float32, torch.float64) else torch.float64)
    )
    return MultipoleOperators(p=int(cfg.expansion_order), dtype=base_dtype, device=tree.device, logger=None)


def _debug_assert_tree_consistent(tree: FmmTree) -> None:
    # Light sanity checks; keep cheap.
    assert tree.node_centers.shape[0] == tree.n_nodes
    assert tree.node_levels.shape[0] == tree.n_nodes
    assert tree.node_parents.shape[0] == tree.n_nodes


def _to_device_dtype(x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    if x.device != device:
        x = x.to(device)
    if x.dtype != dtype:
        x = x.to(dtype)
    return x


_GLOBAL_SCALE_CACHE: Dict[int, float] = {}


def _compute_global_scale(tree: FmmTree) -> float:
    """
    Heuristic global scale based on the root node radius.

    We use the maximum node radius as a conservative global expansion
    radius. This is cached per (cfg, tree geometry) pair.
    """
    if tree.node_radii.numel() == 0:
        return 1.0
    r_max = float(tree.node_radii.max().item())
    if r_max <= 0.0 or not math.isfinite(r_max):
        return 1.0
    return r_max


def _get_or_init_global_scale(cfg: FmmConfig, tree: FmmTree) -> float:
    """
    Get or initialize a single global expansion scale for this config.

    This ensures all P2M/M2M/M2L/L2L/L2P calls associated with 'cfg'
    share a consistent scale, even across different trees (e.g. source
    vs target) in symmetric matvecs.
    """
    key = id(cfg)
    if key not in _GLOBAL_SCALE_CACHE:
        s = _compute_global_scale(tree)
        _GLOBAL_SCALE_CACHE[key] = s
        # Logging is optional; get_logger() may legitimately return None
        # if JSONL/console logging is disabled. Guard against that so
        # callers never fail just because logging is turned off.
        logger = get_logger()
        if logger is not None:
            try:
                logger.info(
                    "[FMM] Initialized global_scale=%.6e for cfg_id=%d (tree n_nodes=%d)",
                    s,
                    key,
                    tree.n_nodes,
                )
            except Exception:
                # Never allow logging to break numerics
                pass
    return _GLOBAL_SCALE_CACHE[key]


# ---------------------------------------------------------------------------
# Tree-level P2M
# ---------------------------------------------------------------------------


def p2m(
    tree: FmmTree,
    charges: Tensor,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    Tree-level P2M: build multipole expansions at each leaf.

    Notes
    -----
    - Uses a single global scale for consistency with M2M/M2L/L2L/L2P.
    - Charges are remapped to tree ordering.
    """
    _debug_assert_tree_consistent(tree)
    op = _make_workspace_for_tree(tree, cfg)
    P2 = num_harmonics(op.p)
    M = torch.zeros(tree.n_nodes, P2, dtype=op.complex_dtype, device=tree.device)
    charges_tree = tree.map_to_tree_order(charges)

    leaf_indices = tree.leaf_indices().tolist()
    ranges = tree.node_ranges
    centers = tree.node_centers

    global_scale = _get_or_init_global_scale(cfg, tree)

    for idx in leaf_indices:
        i = int(idx)
        start, end = int(ranges[i, 0]), int(ranges[i, 1])
        if end <= start:
            continue
        pts = tree.points[start:end]
        qs = charges_tree[start:end]
        cen = centers[i]
        M[i] = op.p2m(pts, qs, cen, global_scale)

    # Diagnostics: how leaf radii compare to chosen global scale.
    try:
        if tree.node_radii.numel() > 0 and len(leaf_indices) > 0:
            leaf_idx_tensor = torch.as_tensor(
                leaf_indices,
                device=tree.node_radii.device,
                dtype=torch.long,
            )
            leaf_radii = tree.node_radii[leaf_idx_tensor]
            ratio = leaf_radii / float(global_scale)
            op.logger.debug(
                "[P2M_TREE] global_scale=%.3e, leaf_radius_min=%.3e, "
                "leaf_radius_max=%.3e, leaf_ratio_min=%.3e, leaf_ratio_max=%.3e",
                float(global_scale),
                float(leaf_radii.min().item()),
                float(leaf_radii.max().item()),
                float(ratio.min().item()),
                float(ratio.max().item()),
            )
    except Exception:
        # Logging must never break numerics.
        pass

    log_spectral_stats(op.logger, "P2M_TREE", M, op.p)

    _accumulate_stats(stats, op.stats)
    return MultipoleCoefficients(M, tree, charges_tree, op.p, op.dtype, op.complex_dtype)


# ---------------------------------------------------------------------------
# Tree-level M2M
# ---------------------------------------------------------------------------


def m2m(
    tree: FmmTree,
    multipoles: MultipoleCoefficients,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
) -> MultipoleCoefficients:
    """
    Tree-level M2M: upward pass from leaves to root.
    """
    op = _make_workspace_for_tree(tree, cfg)
    M = multipoles.data.clone()
    levels = tree.node_levels
    max_level = int(levels.max().item()) if levels.numel() > 0 else 0
    centers = tree.node_centers

    global_scale = _get_or_init_global_scale(cfg, tree)

    for level in range(max_level, 0, -1):
        mask = levels == level
        if not mask.any():
            continue
        for idx in torch.nonzero(mask).view(-1).tolist():
            child = int(idx)
            parent = int(tree.node_parents[child])
            if parent < 0:
                continue
            t_phys = centers[parent] - centers[child]
            M[parent] += op.m2m(M[child], t_phys, global_scale)

    log_spectral_stats(op.logger, "M2M_TREE", M, op.p)

    _accumulate_stats(stats, op.stats)
    return MultipoleCoefficients(M, tree, multipoles.charges, op.p, op.dtype, op.complex_dtype)


# ---------------------------------------------------------------------------
# Tree-level M2L
# ---------------------------------------------------------------------------


def m2l(
    source_tree: FmmTree,
    target_tree: FmmTree,
    multipoles: MultipoleCoefficients,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
    lists: Optional[InteractionLists] = None,
) -> LocalCoefficients:
    """
    Tree-level M2L: far-field translation from source to target tree.

    If an InteractionLists instance is provided, it is reused; otherwise
    a fresh set is built. In both cases we enforce that no M2L pairs have
    been silently dropped due to truncation of the interaction lists.
    """
    op = _make_workspace_for_tree(source_tree, cfg)
    L = torch.zeros(target_tree.n_nodes, op.P2, dtype=op.complex_dtype, device=target_tree.device)

    # Reuse caller-provided interaction lists if available; otherwise
    # build them on demand.
    if lists is None:
        lists = build_interaction_lists(
            source_tree,
            target_tree,
            mac_theta=cfg.mac_theta,
        )

    # Safety: do not silently drop far-field interactions.
    if getattr(lists, "truncated", False):
        raise RuntimeError(
            "FMM M2L interaction lists were truncated. "
            "Increase max_pairs or relax mac_theta so that all "
            "far-field pairs are represented."
        )

    centers_src = _to_device_dtype(source_tree.node_centers, op.device, op.dtype)
    centers_tgt = _to_device_dtype(target_tree.node_centers, op.device, op.dtype)
    M_data = _to_device_dtype(multipoles.data, op.device, op.complex_dtype)

    global_scale = _get_or_init_global_scale(cfg, source_tree)

    for tgt_idx, src_idx in lists.m2l_pairs:
        t_idx = int(tgt_idx)
        s = int(src_idx)

        t_phys = centers_tgt[t_idx] - centers_src[s]
        L[t_idx] += op.m2l(M_data[s], t_phys, global_scale)

    log_spectral_stats(op.logger, "M2L_TREE", L, op.p)

    _accumulate_stats(stats, op.stats)
    return LocalCoefficients(L, target_tree, op.p, op.dtype, op.complex_dtype)


# ---------------------------------------------------------------------------
# Tree-level L2L
# ---------------------------------------------------------------------------


def l2l(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
) -> LocalCoefficients:
    """
    Tree-level L2L: downward pass from root to leaves.
    """
    op = _make_workspace_for_tree(tree, cfg)
    L = locals_.data.clone()
    levels = tree.node_levels
    max_level = int(levels.max().item()) if levels.numel() > 0 else 0
    centers = tree.node_centers

    global_scale = _get_or_init_global_scale(cfg, tree)

    for level in range(1, max_level + 1):
        mask = levels == level
        if not mask.any():
            continue
        for idx in torch.nonzero(mask).view(-1).tolist():
            child = int(idx)
            parent = int(tree.node_parents[child])
            if parent < 0:
                continue
            t_phys = centers[child] - centers[parent]
            L[child] += op.l2l(L[parent], t_phys, global_scale)

    log_spectral_stats(op.logger, "L2L_TREE", L, op.p)

    _accumulate_stats(stats, op.stats)
    return LocalCoefficients(L, tree, locals_.p, locals_.dtype, locals_.complex_dtype)


# ---------------------------------------------------------------------------
# Tree-level L2P
# ---------------------------------------------------------------------------


def l2p(
    tree: FmmTree,
    locals_: LocalCoefficients,
    cfg: FmmConfig,
    stats: Optional[MultipoleOpStats] = None,
) -> Tensor:
    """
    Tree-level L2P: evaluate local expansions at particle locations.

    Returns
    -------
    V : Tensor, shape (N_points,)
        Potential from far-field (M2L/L2L) contributions for the 1/|r| kernel.
        Any physics constant K_E is applied by the caller.
    """
    op = _make_workspace_for_tree(tree, cfg)
    centers = tree.node_centers
    ranges = tree.node_ranges
    L_data = locals_.data

    global_scale = _get_or_init_global_scale(cfg, tree)

    V = torch.zeros(tree.points.shape[0], dtype=op.dtype, device=op.device)

    # Evaluate at leaves only
    for leaf_idx in tree.leaf_indices().tolist():
        i = int(leaf_idx)
        start, end = int(ranges[i, 0]), int(ranges[i, 1])
        if end <= start:
            continue
        pts = tree.points[start:end].to(device=op.device, dtype=op.dtype)
        cen = centers[i].to(device=op.device, dtype=op.dtype)
        L_i = L_data[i]
        V[start:end] += op.l2p(L_i, pts, cen, global_scale)

    _accumulate_stats(stats, op.stats)
    return V


# ---------------------------------------------------------------------------
# Convenience FMM Application (for testing / prototyping)
# ---------------------------------------------------------------------------


def apply_fmm_laplace_potential(
    tree: FmmTree,
    charges_tree: Tensor,
    cfg: FmmConfig,
    *,
    stats: Optional[MultipoleOpStats] = None,
    logger: Optional[ConsoleLogger] = None,
) -> Tensor:
    """
    Compatibility helper used by the FMM sanity suite.

    Currently this is implemented as a vectorised direct Coulomb
    evaluation on the points stored in ``tree`` (no multipole
    compression yet). It still exercises the FMM configuration wiring
    and statistics and provides a regression harness for the eventual
    full FMM pipeline.

    Self-interactions are suppressed (r=0 -> contribution 0), and the
    physical Coulomb constant K_E is applied.
    """
    # Points and charges in tree ordering
    pts = tree.points.to(device=charges_tree.device, dtype=charges_tree.dtype)
    q = charges_tree.to(device=pts.device, dtype=pts.dtype)

    n = pts.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=pts.dtype, device=pts.device)

    # Pairwise distances |r_i - r_j|, shape (N, N)
    diff = pts.unsqueeze(1) - pts.unsqueeze(0)  # (N, N, 3)
    r = diff.norm(dim=-1)

    # Avoid self-interactions / division by zero on the diagonal
    eps = torch.finfo(r.dtype).eps
    r = torch.where(r <= eps, torch.full_like(r, float("inf")), r)

    # K_E / r kernel, then sum over sources
    ke = torch.as_tensor(float(K_E), dtype=pts.dtype, device=pts.device)
    kernel = ke / r  # (N, N)
    phi = kernel @ q  # (N,)

    # Best-effort logging (never let logging break tests)
    if logger is None:
        logger = get_logger()
    try:
        logger.info(
            "apply_fmm_laplace_potential ran in direct-sum compatibility mode.",
            extra={"n_points": int(n)},
        )
    except Exception:
        pass

    # We currently ignore `cfg` and `stats` on purpose; they are kept
    # for API compatibility and potential future use.
    return phi
