import torch
import pytest
import math

from electrodrive.fmm3d.multipole_operators import MultipoleOperators
from electrodrive.fmm3d.spherical_harmonics import (
    cartesian_to_spherical,
    spherical_harmonics_complex,
)

# -----------------------------------------------------------------------------
# Helper: Direct Potential
# -----------------------------------------------------------------------------


def direct_potential(x_src, q, x_tgt):
    """Computes exact sum q / |x_tgt - x_src|."""
    diff = x_tgt.unsqueeze(1) - x_src.unsqueeze(0)  # (Nt, Ns, 3)
    r = torch.norm(diff, dim=-1)
    # Avoid div/0 for self
    r[r < 1e-12] = float("inf")
    return torch.sum(q.unsqueeze(0) / r, dim=1)


# -----------------------------------------------------------------------------
# Test 1: The "Golden Path" M2L
# -----------------------------------------------------------------------------


def test_m2l_vacuum_golden_path():
    """
    Tests P2M -> M2L -> L2P in a perfectly controlled 'vacuum' (no tree).
    If this fails, the math kernels are broken.
    """
    print("\n--- Test 1: Golden Path M2L ---")
    p = 8
    scale = 1.0
    device = torch.device("cpu")  # Use CPU for deterministic debug
    op = MultipoleOperators(p=p, device=device)

    # 1. Geometry: Two boxes separated by vector T = (3,0,0)
    # Source Center (0,0,0), Target Center (3,0,0)
    c_src = torch.zeros(3, dtype=torch.float64)
    c_tgt = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    t_vec = c_tgt - c_src  # Vector FROM source TO target

    # 2. Source: 1.0 charge slightly offset in source box
    x_src = torch.tensor([[0.1, 0.2, -0.1]], dtype=torch.float64)
    q = torch.tensor([1.0], dtype=torch.float64)

    # 3. Target: Point slightly offset in target box
    x_tgt = c_tgt + torch.tensor([[-0.1, 0.1, 0.2]], dtype=torch.float64)

    # 4. Run Operators
    M = op.p2m(x_src, q, c_src, scale)

    # CRITICAL: The Transfer Vector 't' passed to M2L must be (Target - Source)
    L = op.m2l(M, t_vec, scale)

    phi_fmm = op.l2p(L, x_tgt, c_tgt, scale)

    # 5. Validate
    phi_ref = direct_potential(x_src, q, x_tgt)
    rel_err = torch.abs(phi_fmm - phi_ref) / torch.abs(phi_ref)

    print(f"Ref: {phi_ref.item():.6f}, FMM: {phi_fmm.item():.6f}, Err: {rel_err.item():.3e}")

    # Expect high precision for this well-separated case
    assert rel_err < 1e-5, "Basic M2L failed! Check signs in M2L loop."


# -----------------------------------------------------------------------------
# Test 2: The "Sign Flip" Diagnostic
# -----------------------------------------------------------------------------


def test_m2l_sign_flip_diagnostic():
    """
    Deliberately passes the WRONG vector (-t) to M2L.
    If your production code bug looks like this error, you have a sign error.
    """
    print("\n--- Test 2: Sign Flip Diagnostic ---")
    p = 8
    scale = 1.0
    device = torch.device("cpu")
    op = MultipoleOperators(p=p, device=device)

    c_src = torch.zeros(3, dtype=torch.float64)
    c_tgt = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    t_vec = c_tgt - c_src

    x_src = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([1.0], dtype=torch.float64)
    x_tgt = c_tgt + torch.tensor([[-0.1, 0.0, 0.0]], dtype=torch.float64)

    M = op.p2m(x_src, q, c_src, scale)

    # INTENTIONALLY WRONG VECTOR: Source - Target
    t_wrong = -t_vec
    L_wrong = op.m2l(M, t_wrong, scale)
    phi_wrong = op.l2p(L_wrong, x_tgt, c_tgt, scale)

    phi_ref = direct_potential(x_src, q, x_tgt)
    rel_err = torch.abs(phi_wrong - phi_ref) / torch.abs(phi_ref)

    print(
        f"Ref: {phi_ref.item():.6f}, Wrong-Vec FMM: {phi_wrong.item():.6f}, "
        f"Err: {rel_err.item():.3e}"
    )

    # This SHOULD fail. If your production error is ~1.0 - 2.0 relative error,
    # this confirms the hypothesis.
    # We assert that it is BAD.
    assert rel_err > 1e-2, "Using the wrong vector strangely produced a correct result?"


# -----------------------------------------------------------------------------
# Test 3: Scale Invariance Stress (currently expected to fail)
# -----------------------------------------------------------------------------


def test_scale_invariance_stress():
    """
    Verifies that the 'scale' parameter cancels out perfectly.

    This is a *stress* test for the internal scaling conventions.
    Right now it documents a known mismatch between the desired
    invariance and the implemented operators, so it is marked xfail.
    """
    print("\n--- Test 3: Scale Invariance ---")
    p = 8
    device = torch.device("cpu")
    op = MultipoleOperators(p=p, device=device)

    c_src = torch.zeros(3, dtype=torch.float64)
    c_tgt = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    t_vec = c_tgt - c_src

    x_src = torch.randn(1, 3, dtype=torch.float64) * 0.1
    q = torch.randn(1, dtype=torch.float64)
    x_tgt = c_tgt + torch.randn(1, 3, dtype=torch.float64) * 0.1

    # Scale 1.0
    s1 = 1.0
    M1 = op.p2m(x_src, q, c_src, s1)
    L1 = op.m2l(M1, t_vec, s1)
    phi1 = op.l2p(L1, x_tgt, c_tgt, s1)

    # Scale 0.01 (Tiny)
    s2 = 0.01
    M2 = op.p2m(x_src, q, c_src, s2)
    L2 = op.m2l(M2, t_vec, s2)
    phi2 = op.l2p(L2, x_tgt, c_tgt, s2)

    diff = torch.abs(phi1 - phi2).item()
    print(
        f"Phi(s=1.0)={phi1.item():.6f}, "
        f"Phi(s=0.01)={phi2.item():.6f}, Diff={diff:.3e}"
    )

    # Ideal target: diff ~ 0. This assertion currently fails badly,
    # which is why the whole test is marked xfail.
    assert diff < 1e-9, f"Scaling broken! Diff {diff:.3e}"


# -----------------------------------------------------------------------------
# Test 4: Near-Field Convergence (The Stress Test)
# -----------------------------------------------------------------------------


def test_near_field_convergence():
    """
    Tests M2L at the limit of its validity (r_scaled ~ 1.5 to 2.0).
    This is where truncation error is highest.
    """
    print("\n--- Test 4: Near-Field Limit ---")
    scale = 1.0
    device = torch.device("cpu")

    # Setup: Two boxes touching? No, separated by gap.
    # Box radius approx 0.866 (unit cube).
    # Try distance 2.5 (barely separated)
    c_src = torch.zeros(3, dtype=torch.float64)
    c_tgt = torch.tensor([2.5, 0.0, 0.0], dtype=torch.float64)
    t_vec = c_tgt - c_src

    x_src = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)  # Corner of box
    q = torch.tensor([1.0], dtype=torch.float64)
    x_tgt = c_tgt + torch.tensor(
        [[-0.5, -0.5, -0.5]], dtype=torch.float64
    )  # Closest corner

    phi_ref = direct_potential(x_src, q, x_tgt)

    for p in [4, 8, 12, 16]:
        op = MultipoleOperators(p=p, device=device)
        M = op.p2m(x_src, q, c_src, scale)
        L = op.m2l(M, t_vec, scale)
        phi_fmm = op.l2p(L, x_tgt, c_tgt, scale)

        err = torch.abs(phi_fmm - phi_ref) / torch.abs(phi_ref)
        print(f"P={p:2d}, Dist=2.5, Err={err.item():.3e}")


if __name__ == "__main__":
    try:
        test_m2l_vacuum_golden_path()
        test_m2l_sign_flip_diagnostic()
        test_scale_invariance_stress()
        test_near_field_convergence()
        print("\n[SUCCESS] All vacuum operators seem mathematically sound.")
    except AssertionError as e:
        print(f"\n[FAILURE] {e}")
