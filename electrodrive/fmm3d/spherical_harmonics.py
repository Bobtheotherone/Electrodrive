from __future__ import annotations

"""
Core spherical harmonics utilities for the 3D FMM / multipole layer.

Conventions
-----------
Coordinates (physics convention):
    x = r sinθ cosφ
    y = r sinθ sinφ
    z = r cosθ

with:
    θ ∈ [0, π]   polar angle from +z
    φ ∈ [0, 2π)  azimuth from +x towards +y

Complex spherical harmonics (Schmidt Semi-Normalized):
    Y_l^m(θ, φ) = N_l^m P_l^m(cosθ) e^{i m φ}

where:
    - Condon–Shortley phase is included in P_l^m ( (-1)^m ).
    - N_l^m is the Schmidt semi-normalization constant.

    Normalization condition:
        Σ_{m=-l}^{l} Y_l^m(n) Y_l^m(n')^* = P_l(n · n')

    This convention (Schmidt) ensures the Multipole Addition Theorem holds
    without prefactors of 4π/(2l+1), simplifying P2M and L2P operations.

Real spherical harmonics:
    - Built from the complex Y_l^m via standard orthonormal combinations.
    - Compatible with the multipole layer (inherits Schmidt properties).

Design goals
------------
- Pure PyTorch (CPU/GPU) reference implementation for moderate orders.
- Clear (l, m) packing compatible with multipole_operators.
- Workspace and small helper class for caching / repeated evaluations.

Numerical notes
---------------
- Recurrences are implemented in Python loops: fine for l_max ≲ 20–30.
- This module deliberately enforces a *practical* limit:

    l_max <= 32

  to avoid overflow, loss of orthogonality, and NaNs in the Legendre
  recurrences at large order.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "associated_legendre",
    "log_spherical_harmonic_normalization",
    "spherical_harmonics_complex",
    "spherical_harmonics_real",
    "spherical_harmonics",
    "num_harmonics",
    "lm_to_index",
    "index_to_lm",
    "pack_ylm",
    "unpack_ylm",
    "SphericalHarmonicsWorkspace",
    "SphericalHarmonics",
]


# ---------------------------------------------------------------------------
# Constants / limits
# ---------------------------------------------------------------------------

# Practical hard limit for this pure-Python / PyTorch implementation.
# Above this, Legendre recurrences are too unstable and slow.
_L_MAX_HARD_LIMIT = 32

_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


# Optional debug guard for |P_l^m| magnitude.
# By default this is disabled; set the environment variable
# ELECTRODRIVE_LEGENDRE_DEBUG_MAX_ABS to a positive float to enable a
# runtime check on max |P_l^m|. This is intended for diagnostics and
# regression testing, not for normal operation.
_LEGENDRE_DEBUG_MAX_ABS: Optional[float]
try:
    _LEGENDRE_DEBUG_MAX_ABS = float(
        os.getenv("ELECTRODRIVE_LEGENDRE_DEBUG_MAX_ABS", "")
    )
except ValueError:
    _LEGENDRE_DEBUG_MAX_ABS = None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _validate_l_max(l_max: int) -> int:
    """
    Common validation for angular order.

    Enforces a practical limit for this Python implementation.
    """
    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")
    if l_max > _L_MAX_HARD_LIMIT:
        raise ValueError(
            f"Python spherical harmonics limited to l_max <= {_L_MAX_HARD_LIMIT} "
            f"(got {l_max})."
        )
    return l_max


def _require_float32_or_float64(x: Tensor, name: str) -> None:
    """
    Enforce that a tensor is float32 or float64.

    We disallow float16/bfloat16 here because the recurrences are
    numerically delicate and are used in multipole expansions.
    """
    if not torch.is_floating_point(x) or x.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(
            f"{name} must be a floating tensor with dtype float32 or float64, "
            f"got {x.dtype}."
        )


def _ensure_float_tensor(x: Tensor) -> Tensor:
    """
    Ensure x is a floating-point tensor with a sane dtype (float32/float64).

    NOTE: For API-facing functions we call _require_float32_or_float64 first,
    so this helper is primarily used internally where the dtype is already
    known to be acceptable or derived from such inputs.
    """
    if not torch.is_floating_point(x):
        return x.to(torch.float32)
    if x.dtype not in _SUPPORTED_FLOAT_DTYPES:
        return x.to(torch.float32)
    return x


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def cartesian_to_spherical(xyz: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert (..., 3) Cartesian points to (r, theta, phi).

    Physics convention:
        - theta ∈ [0, π] is polar angle from +z.
        - phi ∈ [0, 2π) is azimuth from +x towards +y.

    Parameters
    ----------
    xyz:
        Tensor with last dimension 3.

    Returns
    -------
    r, theta, phi:
        Tensors with shape xyz.shape[:-1].
    """
    xyz = _ensure_float_tensor(xyz)

    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have last dimension 3, got {tuple(xyz.shape)}")

    x, y, z = xyz.unbind(dim=-1)
    r = torch.linalg.norm(xyz, dim=-1)

    finfo = torch.finfo(xyz.dtype)
    eps = finfo.tiny
    safe_r = torch.clamp(r, min=eps)

    # cosθ = z / r, clamped to [-1, 1] for numerical safety
    cos_theta = torch.where(safe_r > 0, z / safe_r, torch.zeros_like(z))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    phi = torch.atan2(y, x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)

    return r, theta, phi


def spherical_to_cartesian(r: Tensor, theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z).

    Uses the same physics convention as :func:`cartesian_to_spherical`.
    Inputs are broadcast to a common shape.

    Returns
    -------
    xyz:
        Tensor of shape broadcast(..., 3).
    """
    r = _ensure_float_tensor(r)
    theta = _ensure_float_tensor(theta)
    phi = _ensure_float_tensor(phi)

    r, theta, phi = torch.broadcast_tensors(r, theta, phi)
    sin_theta = torch.sin(theta)
    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)


# ---------------------------------------------------------------------------
# Associated Legendre polynomials P_l^m(x)
# ---------------------------------------------------------------------------


def associated_legendre(l_max: int, x: Tensor) -> Tensor:
    """
    Compute associated Legendre polynomials P_l^m(x) for 0 <= l <= l_max.

    Parameters
    ----------
    l_max:
        Maximum angular order (must satisfy 0 <= l_max <= 32).
    x:
        cos(theta), with |x| ≤ 1, any broadcastable shape.

    Returns
    -------
    P:
        Tensor with shape x.shape + (l_max + 1, l_max + 1), where
        the last two dims index [l, m] with 0 <= m <= l.
        Entries with m > l are zero.

    Notes
    -----
    - Standard upward recurrences are used:
        * P_0^0 = 1
        * P_l^0 via three-term recurrence
        * diagonals P_m^m via repeated multiplication by sqrt(1 - x^2)
        * off-diagonal terms via standard relation for P_l^m
    - For large l_max and |x| ≈ 1, numerical conditioning degrades;
      this routine is intentionally limited to l_max <= 32.
    """
    _validate_l_max(l_max)

    x = _ensure_float_tensor(x)
    x = torch.clamp(x, -1.0, 1.0)

    *batch_shape, = x.shape
    P = x.new_zeros(*batch_shape, l_max + 1, l_max + 1)

    # l = 0, m = 0
    P[..., 0, 0] = 1.0
    if l_max == 0:
        # Stability check even for trivial case
        if not torch.isfinite(P).all():
            raise RuntimeError("Legendre recurrence became unstable (NaN/inf at l=0).")
        return P

    # m = 0 via three-term recurrence
    P[..., 1, 0] = x
    for l in range(2, l_max + 1):
        P[..., l, 0] = (
            (2.0 * l - 1.0) * x * P[..., l - 1, 0]
            - (l - 1.0) * P[..., l - 2, 0]
        ) / l

    # diagonal P_m^m
    one_minus_x2 = torch.clamp(1.0 - x * x, min=0.0)
    sqrt_one_minus_x2 = torch.sqrt(one_minus_x2)

    if l_max >= 1:
        P[..., 1, 1] = -sqrt_one_minus_x2
    for m in range(2, l_max + 1):
        P[..., m, m] = -(2.0 * m - 1.0) * sqrt_one_minus_x2 * P[..., m - 1, m - 1]

    # P_{m+1}^m
    for m in range(0, l_max):
        P[..., m + 1, m] = (2.0 * m + 1.0) * x * P[..., m, m]

    # general recurrence for l >= m + 2
    # NOTE: For very large l_max this loop would be a hotspot; here it is a
    # reference implementation used up to l_max <= 32.
    for m in range(0, l_max + 1):
        for l in range(m + 2, l_max + 1):
            P[..., l, m] = (
                (2.0 * l - 1.0) * x * P[..., l - 1, m]
                - (l + m - 1.0) * P[..., l - 2, m]
            ) / (l - m)

            # --- DEBUG: INSTANT NAN CHECK ---
            # This is expensive but vital for root cause analysis of the current explosion.
            # It checks immediately after computation if the value is finite.
            if not torch.isfinite(P[..., l, m]).all():
                raise RuntimeError(
                    f"associated_legendre blew up at l={l}, m={m}. "
                    f"Max value found in prev step: {P[..., l-1, m].abs().max()}"
                )
            # --------------------------------

    # ------------------------------------------------------------------
    # Stability checks: detect catastrophic growth / NaNs early
    # ------------------------------------------------------------------
    if not torch.isfinite(P).all():
        # NaN / inf is a hard failure: the recurrence has genuinely blown up.
        raise RuntimeError("Legendre recurrence became unstable (NaN/inf in P_l^m).")

    # Optional debug guard for |P_l^m| magnitude.
    debug_max = _LEGENDRE_DEBUG_MAX_ABS
    if debug_max is not None and P.numel() > 0:
        max_abs = P.abs().max()
        if max_abs > debug_max:
            raise RuntimeError(
                "Legendre recurrence exceeded ELECTRODRIVE_LEGENDRE_DEBUG_MAX_ABS: "
                f"max |P_l^m| = {float(max_abs)} > {float(debug_max)}."
            )

    return P


# ---------------------------------------------------------------------------
# Normalization and (l, m) indexing
# ---------------------------------------------------------------------------


def log_spherical_harmonic_normalization(l: int, m: int) -> float:
    """
    Compute log(N_l^m) using Schmidt Semi-Normalization.

    N_l^m = sqrt( (l-|m|)! / (l+|m|)! )

    This normalization ensures that:
        Sum_{m} Y_l^m(n) Y_l^m(n')^* = P_l(n . n')

    This implies the Multipole Addition Theorem holds without the 4π/(2l+1)
    prefactor found in orthonormal conventions.
    """
    # Use abs(m) to handle -m and m symmetrically
    a = abs(m)

    # log(N) = 0.5 * ( log((l-a)!) - log((l+a)!) )
    # We use lgamma(x) = log((x-1)!), so lgamma(n+1) = log(n!)
    return 0.5 * (
        math.lgamma(l - a + 1.0)
        - math.lgamma(l + a + 1.0)
    )


@lru_cache(maxsize=32)
def _norm_lm_table(l_max: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Normalization constants N_l^m for complex Y_l^m.

    Uses the centralized log_spherical_harmonic_normalization for numerical
    consistency with FMM operators.

    The returned tensor has shape (l_max + 1, l_max + 1) with entries
    N[l, m] defined for 0 <= m <= l and zero otherwise.
    """
    _validate_l_max(l_max)

    N = torch.zeros(l_max + 1, l_max + 1, dtype=dtype, device=device)

    for l in range(l_max + 1):
        for m in range(0, l + 1):
            val = math.exp(log_spherical_harmonic_normalization(l, m))
            N[l, m] = val

    return N


def _compute_alps_fukushima(
    l_max: int,
    x: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Compute *normalized* associated Legendre functions for complex SH.

    This returns
        A_l^m(x) = N_l^m P_l^m(x),

    where P_l^m is the (unnormalized) associated Legendre polynomial with
    Condon–Shortley phase and N_l^m is the Schmidt semi-normalization.
    In particular:

        A_0^0(x) = 1.0  (Schmidt)
        A_0^0(x) = 1 / sqrt(4π) (Orthonormal - NOT USED)

    Parameters
    ----------
    l_max:
        Maximum angular order (0 <= l_max <= 32).
    x:
        cos(theta), broadcastable tensor.
    device, dtype:
        Target device and real dtype (float32 or float64).

    Returns
    -------
    A:
        Tensor with shape x.shape + (l_max + 1, l_max + 1), where the last
        two dims index [l, m] with 0 <= m <= l. Entries with m > l are zero.

    Notes
    -----
    The original Fukushima recurrences operate directly on normalized ALPs
    and require careful bookkeeping of scalar coefficients (g_lm, h_lm, ...).
    Here we implement an algebraically equivalent, but simpler and fully
    batched version by:
        1) computing P_l^m(x) via associated_legendre, and
        2) multiplying by the cached normalization table N_l^m.

    This satisfies the batching and layout requirements for the FMM while
    keeping the implementation compact and easy to verify.
    """
    _validate_l_max(l_max)

    # Ensure x is on the requested device/dtype and in [-1, 1]
    x = x.to(device=device, dtype=dtype)
    x = torch.clamp(x, -1.0, 1.0)

    # Base (unnormalized) P_l^m
    P = associated_legendre(l_max, x)
    P = P.to(device=device, dtype=dtype)

    # Normalization N_l^m (broadcast over batch dimensions)
    N = _norm_lm_table(l_max, dtype=dtype, device=device)  # (l_max+1, l_max+1)
    A = P * N  # broadcasting over the last two dims

    return A


def num_harmonics(l_max: int) -> int:
    """
    Total number of Y_l^m for 0 <= l <= l_max.

    For each l there are (2l + 1) harmonics, so:

        num_harmonics = (l_max + 1)^2
    """
    _validate_l_max(l_max)
    return (l_max + 1) * (l_max + 1)


def lm_to_index(l: int, m: int) -> int:
    """
    Canonical packing index for (l, m): idx = l^2 + (m + l).

    For each fixed l, indices form a contiguous block:

        idx ∈ [l^2, l^2 + 2l] with m ∈ [-l, ..., +l].

    This keeps all entries for a given l grouped and monotone in l.
    """
    if l < 0:
        raise ValueError(f"l must be non-negative, got {l}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got l={l}, m={m}")
    return l * l + (m + l)


def index_to_lm(idx: int) -> Tuple[int, int]:
    """
    Inverse of lm_to_index.

    Recover (l, m) from the canonical index layout.
    """
    if idx < 0:
        raise ValueError(f"idx must be non-negative, got {idx}")
    l = int(math.floor(math.sqrt(idx)))
    m = idx - l * l - l
    if abs(m) > l:
        raise ValueError(f"Invalid packed index {idx}: recovered (l, m)=({l}, {m})")
    return l, m


# ---------------------------------------------------------------------------
# Cached m-values for exp(i m φ)
# ---------------------------------------------------------------------------

_M_VALS_CACHE: Dict[tuple[int, torch.dtype, torch.device], Tensor] = {}
_M_VALS_CACHE_MAX_ENTRIES: int = 64


def _get_m_vals(l_max: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Return tensor [0, 1, ..., l_max] on the given device/dtype, with caching.

    The cache is bounded to avoid unbounded growth when many different
    (l_max, dtype, device) combinations are used.
    """
    key = (int(l_max), dtype, device)
    m_vals = _M_VALS_CACHE.get(key)
    if (
        m_vals is None
        or m_vals.device != device
        or m_vals.dtype != dtype
        or m_vals.numel() != l_max + 1
    ):
        m_vals = torch.arange(0, l_max + 1, device=device, dtype=dtype)
        _M_VALS_CACHE[key] = m_vals
        # Simple bounded cache: drop oldest if over budget.
        if len(_M_VALS_CACHE) > _M_VALS_CACHE_MAX_ENTRIES:
            old_key = next(iter(_M_VALS_CACHE.keys()))
            if old_key != key:
                _M_VALS_CACHE.pop(old_key, None)
    return m_vals


# ---------------------------------------------------------------------------
# Complex and real Y_l^m
# ---------------------------------------------------------------------------


def spherical_harmonics_complex(l_max: int, theta: Tensor, phi: Tensor) -> Tensor:
    """
    Complex spherical harmonics Y_l^m(theta, phi) (Schmidt Semi-Normalized).

    Parameters
    ----------
    l_max:
        Maximum angular order (0 <= l_max <= 32).
    theta, phi:
        Angular coordinates in radians. They are broadcast to a common
        shape S. Both must be float32 or float64 tensors.

    Returns
    -------
    Y:
        Complex tensor with shape S + (l_max + 1, 2*l_max + 1), where
        the last two dims index (l, m_index) with
        m_index = m + l_max, m ∈ [-l_max, ..., +l_max].

    Notes
    -----
    We construct Y_l^m via normalized ALPs:

        A_l^m(cosθ) = N_l^m P_l^m(cosθ)
        Y_l^m(θ, φ) = A_l^m(cosθ) e^{i m φ},

    where N_l^m is the Schmidt semi-normalization and P_l^m includes the
    Condon–Shortley phase. Negative m are filled via

        Y_l^{-m} = (-1)^m conj(Y_l^m).
    """
    _validate_l_max(l_max)
    _require_float32_or_float64(theta, "theta")
    _require_float32_or_float64(phi, "phi")

    theta = _ensure_float_tensor(theta)
    phi = _ensure_float_tensor(phi)

    try:
        theta, phi = torch.broadcast_tensors(theta, phi)
    except RuntimeError as e:
        raise ValueError(
            f"theta and phi must be broadcastable. "
            f"theta.shape={theta.shape}, phi.shape={phi.shape}"
        ) from e

    device = theta.device
    dtype = theta.dtype
    if dtype not in _SUPPORTED_FLOAT_DTYPES:
        # This should be unreachable due to _require_float32_or_float64,
        # but we keep it defensive.
        theta = theta.to(torch.float32)
        phi = phi.to(torch.float32)
        dtype = theta.dtype

    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    batch_shape = theta.shape
    costheta = torch.cos(theta)

    # 1. Normalized ALPs A_l^m(cosθ) = N_l^m P_l^m(cosθ)
    A = _compute_alps_fukushima(l_max, costheta, device=device, dtype=dtype)
    # A has shape batch_shape + (l_max + 1, l_max + 1)

    # 2. Combine with azimuthal factor e^{i m φ}
    Y = torch.zeros(
        *batch_shape,
        l_max + 1,
        2 * l_max + 1,
        dtype=complex_dtype,
        device=device,
    )

    phi_complex = phi.to(complex_dtype)

    for m in range(l_max + 1):
        # e^{i m φ}, shape batch_shape
        if m == 0:
            exp_imphi = torch.ones_like(phi_complex, dtype=complex_dtype)
        else:
            exp_imphi = torch.exp(1j * m * phi_complex)

        # Positive m (including m = 0): Y_l^m = A_l^m * e^{i m φ}
        # A[..., :, m] has shape batch_shape + (l_max + 1,)
        Y[..., :, l_max + m] = A[..., :, m].to(complex_dtype) * exp_imphi[..., None]

        if m > 0:
            # Negative m via Condon–Shortley symmetry:
            #   Y_l^{-m} = (-1)^m conj(Y_l^m)
            sign = -1.0 if (m % 2) != 0 else 1.0
            Y[..., :, l_max - m] = sign * torch.conj(Y[..., :, l_max + m])

    return Y


def spherical_harmonics_real(l_max: int, theta: Tensor, phi: Tensor) -> Tensor:
    """
    Real spherical harmonics constructed from complex Y_l^m.

    Parameters
    ----------
    l_max:
        Maximum angular order (0 <= l_max <= 32).
    theta, phi:
        Angular coordinates in radians (physics convention).
        Both must be float32 or float64 tensors.

    Returns
    -------
    Yr:
        Real tensor with shape S + (l_max + 1, 2*l_max + 1), same layout
        as complex version.

    Mapping (per l, m>0)
    --------------------
    Let m_index = m + l_max, and Y^c the complex basis (Schmidt):

        Yr[l, m_index]   = sqrt(2) (-1)^m Re(Y^c_{l, m})
        Yr[l, -m_index]  = sqrt(2) (-1)^m Im(Y^c_{l, m})
        Yr[l, m0_index]  = Re(Y^c_{l, 0})

    Since Y^c is Schmidt semi-normalized, Yr satisfies the addition
    theorem P_l(n . n') = sum_m Yr(n) Yr(n') without prefactors.
    """
    _validate_l_max(l_max)
    _require_float32_or_float64(theta, "theta")
    _require_float32_or_float64(phi, "phi")

    Yc = spherical_harmonics_complex(l_max, theta, phi)

    *batch_shape, L, M = Yc.shape
    if L != l_max + 1 or M != 2 * l_max + 1:
        raise ValueError(
            f"Internal shape error: complex Y must have shape "
            f"(..., {l_max + 1}, {2 * l_max + 1}), got {tuple(Yc.shape)}"
        )

    device = Yc.device
    real_dtype = torch.float32 if Yc.dtype == torch.complex64 else torch.float64

    Yr = torch.zeros(*batch_shape, L, M, dtype=real_dtype, device=device)

    # m = 0
    m0_index = l_max
    Yr[..., :, m0_index] = Yc[..., :, m0_index].real.to(real_dtype)

    sqrt2 = math.sqrt(2.0)
    for m in range(1, l_max + 1):
        idx_pos = m + l_max
        idx_neg = -m + l_max
        Ym = Yc[..., :, idx_pos]
        factor = ((-1.0) ** m) * sqrt2
        Yr[..., :, idx_pos] = (factor * Ym.real).to(real_dtype)
        Yr[..., :, idx_neg] = (factor * Ym.imag).to(real_dtype)

    return Yr


def spherical_harmonics(
    l_max: int,
    theta: Tensor,
    phi: Tensor,
    *,
    kind: str = "real",
) -> Tensor:
    """
    Dispatch to real or complex spherical harmonics.

    Parameters
    ----------
    l_max:
        Maximum angular order (0 <= l_max <= 32).
    theta, phi:
        Angular coordinates (radians, physics convention).
        Both must be float32 or float64 tensors.
    kind:
        "real" or "complex".

    Returns
    -------
    Y:
        Tensor with shape S + (l_max + 1, 2*l_max + 1).
    """
    if kind == "real":
        return spherical_harmonics_real(l_max, theta, phi)
    if kind == "complex":
        return spherical_harmonics_complex(l_max, theta, phi)
    raise ValueError(f"kind must be 'real' or 'complex', got {kind!r}")


# ---------------------------------------------------------------------------
# Packing / unpacking of Y_l^m values
# ---------------------------------------------------------------------------


def pack_ylm(Y: Tensor, l_max: int) -> Tensor:
    """
    Pack Y_l^m into a flat vector with canonical lm_to_index layout.

    Parameters
    ----------
    Y:
        Tensor with shape (..., l_max + 1, 2*l_max + 1), typically the
        output of spherical_harmonics_real/complex.
    l_max:
        Maximum angular order.

    Returns
    -------
    coeffs:
        Tensor with shape (..., num_harmonics(l_max)), where the last
        dimension runs over (l, m) via lm_to_index(l, m).
    """
    _validate_l_max(l_max)

    *batch_shape, L, M = Y.shape
    if L != l_max + 1 or M != 2 * l_max + 1:
        raise ValueError(
            f"Y must have shape (..., {l_max + 1}, {2 * l_max + 1}), got {tuple(Y.shape)}"
        )

    coeffs = Y.new_zeros(*batch_shape, num_harmonics(l_max))

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = lm_to_index(l, m)
            m_index = m + l_max
            coeffs[..., idx] = Y[..., l, m_index]

    return coeffs


def unpack_ylm(coeffs: Tensor, l_max: int) -> Tensor:
    """
    Inverse of pack_ylm.

    Given packed coefficients with canonical lm_to_index layout,
    reconstruct the (l, m) grid Y_l^m.

    Parameters
    ----------
    coeffs:
        Tensor with shape (..., num_harmonics(l_max)).
    l_max:
        Maximum angular order.

    Returns
    -------
    Y:
        Tensor with shape (..., l_max + 1, 2*l_max + 1).
    """
    _validate_l_max(l_max)

    *batch_shape, K = coeffs.shape
    expected = num_harmonics(l_max)
    if K != expected:
        raise ValueError(f"coeffs must have last dimension {expected}, got {K}")

    Y = coeffs.new_zeros(*batch_shape, l_max + 1, 2 * l_max + 1)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = lm_to_index(l, m)
            m_index = m + l_max
            Y[..., l, m_index] = coeffs[..., idx]

    return Y


# ---------------------------------------------------------------------------
# Workspace / caching for repeated grids
# ---------------------------------------------------------------------------


@dataclass
class SphericalHarmonicsWorkspace:
    """
    Helper for repeated spherical-harmonics evaluations on a fixed grid.

    - Cache is exact-match: theta/phi must match in shape, dtype, device,
      and values (torch.equal).
    - Useful for reusing quadrature / interpolation grids across solves.
    """

    l_max: int
    kind: str = "real"
    cached_theta: Optional[Tensor] = None
    cached_phi: Optional[Tensor] = None
    cached_Y: Optional[Tensor] = None

    def __post_init__(self) -> None:
        _validate_l_max(self.l_max)
        if self.kind not in ("real", "complex"):
            raise ValueError(
                f"SphericalHarmonicsWorkspace.kind must be 'real' or 'complex', "
                f"got {self.kind!r}."
            )

    # ---- core API ---------------------------------------------------------

    def eval(self, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Evaluate Y_l^m(theta, phi) in the configured basis (no caching).

        If a cached grid exists and matches exactly, it is returned;
        otherwise Y_l^m is computed without modifying the cache.
        """
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        cached = self.maybe_get_cached(theta, phi)
        if cached is not None:
            return cached
        return spherical_harmonics(self.l_max, theta, phi, kind=self.kind)

    def eval_cached(self, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Evaluate Y_l^m with automatic caching.

        If a matching grid is cached, it is returned; otherwise Y_l^m
        is computed and stored for future calls.
        """
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        cached = self.maybe_get_cached(theta, phi)
        if cached is not None:
            return cached
        return self.cache_grid(theta, phi)

    def eval_from_cartesian(
        self,
        xyz: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert xyz to (r, theta, phi) and evaluate Y_l^m.

        Returns
        -------
        r, theta, phi, Y:
            Radial distances, angles, and harmonics evaluated on the grid.
        """
        r, theta, phi = cartesian_to_spherical(xyz)
        # cartesian_to_spherical already enforces floating dtype; but for
        # clarity we still enforce the angular dtype contract here.
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        Y = self.eval(theta, phi)
        return r, theta, phi, Y

    def eval_packed(self, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Evaluate Y_l^m and pack into flat coefficient vectors.
        """
        Y = self.eval(theta, phi)
        return pack_ylm(Y, self.l_max)

    # ---- cache management -------------------------------------------------

    def cache_grid(self, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Precompute Y_l^m on a grid and store inside the workspace.

        Subsequent eval_cached calls with the same grid will reuse it.
        """
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        Y = spherical_harmonics(self.l_max, theta, phi, kind=self.kind)
        self.cached_theta = theta
        self.cached_phi = phi
        self.cached_Y = Y
        return Y

    def maybe_get_cached(self, theta: Tensor, phi: Tensor) -> Optional[Tensor]:
        """
        Return cached Y_l^m if the grid matches exactly, else None.

        Matching criteria:
            - same device and dtype
            - same shape
            - torch.equal on theta, phi
        """
        if self.cached_Y is None:
            return None
        if self.cached_theta is None or self.cached_phi is None:
            return None

        if (
            theta.device != self.cached_theta.device
            or phi.device != self.cached_phi.device
            or theta.dtype != self.cached_theta.dtype
            or phi.dtype != self.cached_phi.dtype
        ):
            return None

        if (
            theta.shape == self.cached_theta.shape
            and phi.shape == self.cached_phi.shape
            and torch.equal(theta, self.cached_theta)
            and torch.equal(phi, self.cached_phi)
        ):
            return self.cached_Y

        return None

    def clear_cache(self) -> None:
        """
        Explicitly drop any cached grid and harmonics.

        This is useful to release memory when large grids are no longer needed.
        """
        self.cached_theta = None
        self.cached_phi = None
        self.cached_Y = None


# ---------------------------------------------------------------------------
# SphericalHarmonics helper class (used by kernels_cpu, etc.)
# ---------------------------------------------------------------------------


@dataclass
class SphericalHarmonics:
    """
    Small helper for per-point spherical harmonics evaluations.

    This is the API expected by kernels such as p2m_cpu / l2p_cpu:

        sh = SphericalHarmonics(p)          # default: real basis
        Y = sh.compute(xyz)                 # shape (N, (p+1)^2) packed

    By default, `compute` returns *packed* coefficients using the canonical
    lm_to_index layout. If you need the full (l, m) grid, use
    `compute(xyz, packed=False)`.

    Parameters
    ----------
    l_max:
        Maximum angular order.
    kind:
        "real" (default) or "complex".
    """

    l_max: int
    kind: str = "real"

    def __post_init__(self) -> None:
        self.l_max = _validate_l_max(self.l_max)
        if self.kind not in ("real", "complex"):
            raise ValueError(
                f"SphericalHarmonics.kind must be 'real' or 'complex', got {self.kind!r}."
            )

    # ------------------------- public API ---------------------------------

    @property
    def n_coeffs(self) -> int:
        """Total number of packed coefficients = (l_max + 1)^2."""
        return num_harmonics(self.l_max)

    def compute(
        self,
        xyz: Tensor,
        *,
        packed: bool = True,
    ) -> Tensor:
        """
        Evaluate spherical harmonics at Cartesian points.

        Parameters
        ----------
        xyz:
            Tensor of shape (..., 3).
        packed:
            If True (default), return packed coefficients with shape
            (..., (l_max + 1)^2). If False, return the full grid
            (..., l_max + 1, 2*l_max + 1).

        Returns
        -------
        Y:
            Packed or grid representation as described above.
        """
        xyz = _ensure_float_tensor(xyz)
        if xyz.shape[-1] != 3:
            raise ValueError(
                f"SphericalHarmonics.compute expects last dimension 3, got {tuple(xyz.shape)}"
            )

        orig_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(-1, 3)

        _, theta, phi = cartesian_to_spherical(xyz_flat)
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        Y_grid = spherical_harmonics(self.l_max, theta, phi, kind=self.kind)
        if not packed:
            return Y_grid.reshape(*orig_shape, self.l_max + 1, 2 * self.l_max + 1)

        Y_packed = pack_ylm(Y_grid, self.l_max)
        return Y_packed.reshape(*orig_shape, self.n_coeffs)

    def compute_from_angles(
        self,
        theta: Tensor,
        phi: Tensor,
        *,
        packed: bool = True,
    ) -> Tensor:
        """
        Evaluate spherical harmonics given (theta, phi) directly.

        Parameters
        ----------
        theta, phi:
            Angular coordinates in radians, broadcastable to a common shape.
        packed:
            If True (default), return packed coefficients; else full grid.

        Returns
        -------
        Y:
            Packed or grid representation as described above.
        """
        _require_float32_or_float64(theta, "theta")
        _require_float32_or_float64(phi, "phi")

        theta, phi = torch.broadcast_tensors(theta, phi)
        orig_shape = theta.shape

        Y_grid = spherical_harmonics(self.l_max, theta, phi, kind=self.kind)
        if not packed:
            return Y_grid.reshape(*orig_shape, self.l_max + 1, 2 * self.l_max + 1)

        Y_packed = pack_ylm(Y_grid, self.l_max)
        return Y_packed.reshape(*orig_shape, self.n_coeffs)