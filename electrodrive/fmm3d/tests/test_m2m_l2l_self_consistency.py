import math
import torch
import pytest

from electrodrive.fmm3d.spherical_harmonics import (
    cartesian_to_spherical,
    spherical_harmonics_complex,
    pack_ylm,
    index_to_lm,
)


# -----------------------------------------------------------------------------
# Helper: Direct potential
# -----------------------------------------------------------------------------

def _direct_potential(x_src: torch.Tensor, q: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    """
    Direct 1/r potential from point charges.
    x_src: (Ns, 3), q: (Ns,), x_tgt: (Nt, 3)
    Returns: (Nt,) real tensor
    """
    diff = x_tgt[:, None, :] - x_src[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=-1)
    return torch.sum(q[None, :] / r, dim=1)


# -----------------------------------------------------------------------------
# Helper: Multipole expansion (Spec Implementation - Schmidt Normalized)
# -----------------------------------------------------------------------------

def _p2m_packed(
    x_src: torch.Tensor,
    q: torch.Tensor,
    center: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Compute multipole coefficients M_lm (packed) for sources at x_src.
    
    Schmidt Convention:
      M_lm = sum_j q_j * rho_j^l * conj(Y_lm(Omega_j))
    """
    x_rel = x_src - center
    r, theta, phi = cartesian_to_spherical(x_rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    
    Y_packed = pack_ylm(Y, p)  # (Ns, P2)
    
    P2 = (p + 1) ** 2
    M = torch.zeros(P2, dtype=torch.complex128, device=x_src.device)
    q_c = q.to(dtype=M.dtype)

    for idx in range(P2):
        l, m = index_to_lm(idx)
        rad = r ** l
        # Accumulate: q * rho^l * conj(Y)
        term = q_c * rad * torch.conj(Y_packed[:, idx])
        M[idx] = torch.sum(term)
        
    return M


def _eval_multipole_potential(
    x_tgt: torch.Tensor,
    center: torch.Tensor,
    M: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Evaluate potential at x_tgt due to multipole coefficients M_lm.

    Schmidt Convention (No 4pi/(2l+1) factor):
      phi(r) = sum_{l,m} (M_lm / r^{l+1}) * Y_lm(Omega_r)
    """
    x_rel = x_tgt - center
    r, theta, phi = cartesian_to_spherical(x_rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    Y_packed = pack_ylm(Y, p) # (Nt, P2)

    P2 = (p + 1) ** 2
    phi_val = torch.zeros(x_tgt.shape[0], dtype=torch.complex128, device=x_tgt.device)

    for idx in range(P2):
        l, m = index_to_lm(idx)
        
        # Radial term: 1 / r^{l+1}
        rad_inv = r ** (-(l + 1.0))
        
        # No prefactor for Schmidt!
        term = M[idx] * rad_inv * Y_packed[:, idx]
        phi_val += term

    return phi_val.real


# -----------------------------------------------------------------------------
# Helper: Local expansion (Spec Implementation - Schmidt Normalized)
# -----------------------------------------------------------------------------

def _p2local_packed(
    x_src: torch.Tensor,
    q: torch.Tensor,
    center: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Compute local expansion coefficients L_lm (packed).
    
    Schmidt Convention:
      L_lm = sum_j q_j * conj(Y_lm(Omega_j)) / rho_j^{l+1}
    """
    x_rel = x_src - center
    rho, theta, phi = cartesian_to_spherical(x_rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    Y_packed = pack_ylm(Y, p)

    P2 = (p + 1) ** 2
    L = torch.zeros(P2, dtype=torch.complex128, device=x_src.device)
    q_c = q.to(dtype=L.dtype)

    for idx in range(P2):
        l, m = index_to_lm(idx)
        
        rad_inv = rho ** (-(l + 1.0))
        
        # Accumulate: q * (1/rho^{l+1}) * conj(Y)
        term = q_c * rad_inv * torch.conj(Y_packed[:, idx])
        L[idx] = torch.sum(term)

    return L


def _eval_local_potential(
    x_tgt: torch.Tensor,
    center: torch.Tensor,
    L: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Evaluate potential at x_tgt due to local coefficients L_lm.

    Schmidt Convention:
      phi(r) = sum_{l,m} L_lm * r^l * Y_lm(Omega_r)
    """
    x_rel = x_tgt - center
    r, theta, phi = cartesian_to_spherical(x_rel)
    Y = spherical_harmonics_complex(p, theta, phi)
    Y_packed = pack_ylm(Y, p)

    P2 = (p + 1) ** 2
    phi_val = torch.zeros(x_tgt.shape[0], dtype=torch.complex128, device=x_tgt.device)

    for idx in range(P2):
        l, m = index_to_lm(idx)
        
        rad = r ** l
        term = L[idx] * rad * Y_packed[:, idx]
        phi_val += term

    return phi_val.real


# -----------------------------------------------------------------------------
# 1) Multipole recentering check
# -----------------------------------------------------------------------------

def test_multipole_recenter_far_field_consistency():
    torch.manual_seed(1234)
    p = 6
    n_src = 80
    n_tgt = 64

    shift = 0.30
    R_src = 0.20
    R_tgt = shift + R_src + 1.0

    c_parent = torch.zeros(3, dtype=torch.float64)
    c_child = torch.tensor([shift, 0.0, 0.0], dtype=torch.float64)

    # Sources
    x = torch.randn(n_src, 3, dtype=torch.float64)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    radii = torch.rand(n_src, 1, dtype=torch.float64) * R_src
    x_src = c_child + x * radii
    q = torch.randn(n_src, dtype=torch.float64)

    # Targets
    x = torch.randn(n_tgt, 3, dtype=torch.float64)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    x_tgt = x * R_tgt

    # Expansions
    M_child = _p2m_packed(x_src, q, c_child, p)
    M_parent = _p2m_packed(x_src, q, c_parent, p)

    phi_child = _eval_multipole_potential(x_tgt, c_child, M_child, p)
    phi_parent = _eval_multipole_potential(x_tgt, c_parent, M_parent, p)
    phi_ref = _direct_potential(x_src, q, x_tgt)

    rel_child = torch.linalg.norm(phi_child - phi_ref) / torch.linalg.norm(phi_ref)
    rel_parent = torch.linalg.norm(phi_parent - phi_ref) / torch.linalg.norm(phi_ref)
    rel_diff = torch.linalg.norm(phi_child - phi_parent) / torch.linalg.norm(phi_parent)

    print(f"\n[M2M Check] Rel Errors -> Child: {rel_child:.3e}, Parent: {rel_parent:.3e}, Diff: {rel_diff:.3e}")

    # With correct spec logic, these should be very small (~1e-5)
    assert rel_child < 1e-4
    assert rel_parent < 1e-4
    assert rel_diff < 1e-4


# -----------------------------------------------------------------------------
# 2) Local recentering check
# -----------------------------------------------------------------------------

def test_local_recenter_near_field_consistency():
    torch.manual_seed(4321)
    
    # UPDATED: Increased p from 6 to 8 to handle L2L shift accuracy
    p = 8 
    
    n_src = 80
    n_tgt = 128

    shift = 0.25
    R_sources = 2.0
    R_dom = 0.5

    c_parent = torch.zeros(3, dtype=torch.float64)
    c_child = torch.tensor([shift, 0.0, 0.0], dtype=torch.float64)

    # Sources (far field)
    x = torch.randn(n_src, 3, dtype=torch.float64)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    radii = R_sources + 0.2 * torch.rand(n_src, 1, dtype=torch.float64)
    x_src = x * radii
    q = torch.randn(n_src, dtype=torch.float64)

    # Targets (local)
    x = torch.randn(n_tgt, 3, dtype=torch.float64)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    radii = torch.rand(n_tgt, 1, dtype=torch.float64) * R_dom
    x_tgt = x * radii

    # Expansions
    L_parent = _p2local_packed(x_src, q, c_parent, p)
    L_child = _p2local_packed(x_src, q, c_child, p)

    phi_parent = _eval_local_potential(x_tgt, c_parent, L_parent, p)
    phi_child = _eval_local_potential(x_tgt, c_child, L_child, p)
    phi_ref = _direct_potential(x_src, q, x_tgt)

    rel_parent = torch.linalg.norm(phi_parent - phi_ref) / torch.linalg.norm(phi_ref)
    rel_child = torch.linalg.norm(phi_child - phi_ref) / torch.linalg.norm(phi_ref)
    rel_diff = torch.linalg.norm(phi_child - phi_parent) / torch.linalg.norm(phi_parent)

    print(f"\n[L2L Check] Rel Errors -> Parent: {rel_parent:.3e}, Child: {rel_child:.3e}, Diff: {rel_diff:.3e}")

    assert rel_parent < 1e-4
    assert rel_child < 1e-4
    assert rel_diff < 1e-4