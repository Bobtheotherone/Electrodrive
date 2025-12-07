from __future__ import annotations

import torch
import pytest
import time
from electrodrive.fmm3d.multipole_operators import MultipoleOperators, p2m, m2l, l2p
from electrodrive.fmm3d.spherical_harmonics import num_harmonics

# -----------------------------------------------------------------------------
# INSTANT DIAGNOSTIC TEST
# -----------------------------------------------------------------------------

def test_isolated_m2l_operator():
    """
    MICRO-BENCHMARK: Tests the M2L operator in isolation.
    
    Why this is fast:
    - No tree construction.
    - No interaction lists.
    - Runs the heavy m2l loops EXACTLY ONCE.
    
    Why this catches the bug:
    - Uses a diagonal translation vector (3, 2, 1) to trigger all (l,m) phase terms.
    - Uses p=4 to ensure complex harmonics are active.
    - If M2L has sign/conjugation errors, this will fail with ~30% error.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    p = 4  # Sufficient to trigger complex harmonic bugs, fast enough for Python
    
    # 1. Setup Geometry
    # Source cluster at origin, Target cluster shifted diagonally
    center_src = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    center_tgt = torch.tensor([3.0, 2.0, 1.0], dtype=dtype, device=device) # Asymmetric shift
    translation = center_tgt - center_src
    dist = torch.norm(translation).item()
    
    # Sources: small random cloud around origin
    # Targets: small random cloud around target center
    # Ensure separation > radius sum for valid multipole convergence
    torch.manual_seed(42)
    src_points = (torch.rand(10, 3, dtype=dtype, device=device) - 0.5) * 0.5 # radius ~0.4
    tgt_points = center_tgt + (torch.rand(10, 3, dtype=dtype, device=device) - 0.5) * 0.5
    
    charges = torch.randn(10, dtype=dtype, device=device)
    
    # 2. Initialize Operators
    # Use scale=1.0 for simplicity to test the raw translation logic
    scale = 1.0 
    op = MultipoleOperators(p=p, dtype=dtype, device=device)
    
    # 3. FMM Path: P2M -> M2L -> L2P
    # Step A: P2M (Source -> Multipole at Source Center)
    M_packed = op.p2m(src_points, charges, center_src, scale)
    
    # Step B: M2L (Multipole at Source -> Local at Target Center)
    # This is the critical step that was buggy
    L_packed = op.m2l(M_packed, translation, scale)
    
    # Step C: L2P (Local at Target -> Potential at Target Points)
    V_fmm = op.l2p(L_packed, tgt_points, center_tgt, scale)
    
    # 4. Reference Path: Direct Summation
    # V = Sum q_i / |r_tgt - r_src|
    # (Ignoring K_E constant for relative error check, or assuming K_E=1 in logic if consistent)
    # Note: Your repo uses K_E in apply_fmm_laplace_potential, but operators usually return raw sums.
    # Let's check pure 1/r sum.
    diff = tgt_points.unsqueeze(1) - src_points.unsqueeze(0) # [Nt, Ns, 3]
    dists = torch.norm(diff, dim=2)
    V_direct = torch.sum(charges.unsqueeze(0) / dists, dim=1)
    
    # 5. Check Error
    # NOTE: The operators usually assume internal scaling. 
    # If your l2p output needs multiplying by K_E, we adjust. 
    # But usually ratio is robust.
    
    # Calculate relative error
    num = torch.norm(V_fmm - V_direct).item()
    den = torch.norm(V_direct).item()
    rel_err = num / den
    
    print(f"\n[ISOLATED OP TEST] p={p}, dist={dist:.2f}")
    print(f"Direct Norm: {den:.4e}")
    print(f"FMM Norm:    {torch.norm(V_fmm).item():.4e}")
    print(f"Rel Error:   {rel_err:.4e}")
    
    # Pass criteria: 
    # p=4 at distance ~3.7 (approx 3.7/0.5 ratio) should be around 1e-3 or 1e-4.
    # The BUG produces errors ~ 3e-1 (30%).
    if rel_err > 1e-2:
        pytest.fail(f"M2L Operator Broken! Error {rel_err:.2e} is too high.")
    else:
        print("SUCCESS: M2L Operator logic is sound.")


def test_tiny_integration():
    """
    Tiny integration test to verify P2M->M2M->M2L->L2L->L2P chain works together.
    Uses extreme depth but very few points.
    """
    from electrodrive.fmm3d.bem_fmm import make_laplace_fmm_backend
    
    # 50 points, max_leaf=5 -> Depth ~3
    # p=3 for speed
    N = 50
    dtype = torch.float64
    device = torch.device("cpu")
    
    x = torch.rand(N, 3, dtype=dtype, device=device)
    q = torch.randn(N, dtype=dtype, device=device)
    areas = torch.ones_like(q)
    
    fmm = make_laplace_fmm_backend(
        src_centroids=x,
        areas=areas,
        max_leaf_size=5,
        expansion_order=3,
        theta=0.5
    )
    
    t0 = time.perf_counter()
    res = fmm.matvec(
        sigma=q,
        src_centroids=x,
        areas=areas,
        tile_size=1024,
        self_integrals=None,
    )
    t_fmm = time.perf_counter() - t0
    
    # Direct
    # standard 1/r sum
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    r = diff.norm(dim=-1) + torch.eye(N, device=device)*1e9
    ref = (q.unsqueeze(0) / r).sum(dim=1) * 8.987551787e9 # K_E
    
    err = torch.norm(res - ref) / torch.norm(ref)
    print(f"\n[TINY INTEGRATION] Err={err:.3e}, Time={t_fmm*1000:.1f}ms")
    
    assert err < 5e-2