import sys
import os
import torch
import math

# Ensure we load the local code, not an installed version
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from electrodrive.fmm3d.multipole_operators import (
        MultipoleOperators, 
        _rescale_multipoles_packed, 
        _rescale_locals_packed
    )
    from electrodrive.fmm3d.spherical_harmonics import index_to_lm
except ImportError as e:
    print("\n[CRITICAL] Could not import electrodrive modules.")
    print(f"Current path: {sys.path}")
    print("Ensure you run this from the repo root: 'python audit_kernels.py'")
    raise e

# --- Helpers ---
GRN = "\033[92m"
RED = "\033[91m"
YEL = "\033[93m"
RST = "\033[0m"

def report(name, err, threshold=1e-12, details=""):
    passed = err < threshold
    status = f"{GRN}PASS{RST}" if passed else f"{RED}FAIL{RST}"
    print(f"[{status}] {name:<30} | RelErr: {err:.2e} {details}")
    return passed

def spectral_audit(v_test, v_ref, p, name):
    """Analyzes error per degree l to find where the math breaks."""
    print(f"    {YEL}>>> Spectral Audit for {name}:{RST}")
    v_diff = v_test - v_ref
    l_map = []
    for idx in range(v_test.numel()):
        l, m = index_to_lm(idx)
        l_map.append(l)
    l_map = torch.tensor(l_map, device=v_test.device)
    
    for l in range(p + 1):
        mask = (l_map == l)
        if not mask.any():
            continue
        norm_ref = torch.norm(v_ref[mask])
        norm_diff = torch.norm(v_diff[mask])
        rel = norm_diff / norm_ref if norm_ref > 1e-15 else norm_diff
        scale_status = ""
        if rel > 1e-5:
            dot = torch.sum(v_test[mask] * torch.conj(v_ref[mask])).real
            ref_sq = torch.sum(v_ref[mask] * torch.conj(v_ref[mask])).real
            alpha = dot / ref_sq
            scale_status = f"| Bias factor alpha ~ {alpha:.4f}"
        print(f"      l={l}: RelErr={rel:.2e} {scale_status}")

# --- Tests ---

def test_rescaling_logic(op):
    print("\n=== TEST 1: Rescaling Formulas (S_from -> S_to) ===")
    scale1, scale2 = 0.5, 1.2
    center = torch.zeros(3, dtype=op.dtype, device=op.device)
    src = torch.randn(5, 3, dtype=op.dtype, device=op.device) * 0.1
    q = torch.randn(5, dtype=op.dtype, device=op.device)

    M1 = op.p2m(src, q, center, scale1)
    M2_ref = op.p2m(src, q, center, scale2)
    ratio = scale1 / scale2
    M2_test = _rescale_multipoles_packed(M1, ratio, op.p)
    err = torch.norm(M2_test - M2_ref) / torch.norm(M2_ref)
    if not report("P2M Rescaling (ratio^l)", err):
        spectral_audit(M2_test, M2_ref, op.p, "P2M Rescale")

    t_vec = torch.tensor([2.0, 0.0, 0.0], dtype=op.dtype, device=op.device)
    L1 = op.m2l(M1, t_vec * (1.0/scale1), scale1)
    L2_ref = op.m2l(M2_ref, t_vec * (1.0/scale2), scale2)
    ratio_L = scale2 / scale1 
    L2_test = _rescale_locals_packed(L1, ratio_L, op.p)
    err_L = torch.norm(L2_test - L2_ref) / torch.norm(L2_ref)
    if not report("Local Rescaling (ratio^l+1)", err_L):
        spectral_audit(L2_test, L2_ref, op.p, "Local Rescale")

def test_translation_fixed_scale(op):
    print("\n=== TEST 2: Translations (Schmidt Normalized, Fixed Scale) ===")
    scale = 1.0
    child_cen = torch.zeros(3, dtype=op.dtype, device=op.device)
    shift = torch.tensor([0.1, 0.1, 0.1], dtype=op.dtype, device=op.device)
    parent_cen = child_cen + shift
    src = torch.randn(10, 3, dtype=op.dtype, device=op.device) * 0.05
    q = torch.randn(10, dtype=op.dtype, device=op.device)
    
    M_child = op.p2m(src, q, child_cen, scale)
    M_parent_ref = op.p2m(src, q, parent_cen, scale)
    M_parent_test = op.m2m(M_child, shift, scale)
    err_m2m = torch.norm(M_parent_test - M_parent_ref) / torch.norm(M_parent_ref)
    if not report("M2M Translation", err_m2m):
        spectral_audit(M_parent_test, M_parent_ref, op.p, "M2M")

    src_cen = torch.tensor([5.0, 0.0, 0.0], dtype=op.dtype, device=op.device)
    M_remote = op.p2m(src + src_cen, q, src_cen, scale)
    L_parent = op.m2l(M_remote, parent_cen - src_cen, scale)
    L_child_ref = op.m2l(M_remote, child_cen - src_cen, scale)
    L_child_test = op.l2l(L_parent, child_cen - parent_cen, scale)
    err_l2l = torch.norm(L_child_test - L_child_ref) / torch.norm(L_child_ref)
    if not report("L2L Translation", err_l2l):
        spectral_audit(L_child_test, L_child_ref, op.p, "L2L")

def test_l2p_bias(op):
    print("\n=== TEST 3: L2P Global Bias Check (pure 1/r) ===")
    scale = 0.45
    center = torch.zeros(3, dtype=op.dtype, device=op.device)

    # Remote single charge at distance ~10
    src_remote = torch.tensor([[10.0, 0.0, 0.0]], dtype=op.dtype, device=op.device)
    q = torch.tensor([1.0], dtype=op.dtype, device=op.device)

    # Build remote multipole about src_remote with scale 1.0
    M_remote = op.p2m(src_remote, q, src_remote, 1.0)

    # M2L to a local at the origin using dimensionless translation
    t_vec = center - src_remote[0]
    L_packed = op.m2l(M_remote, t_vec * (1.0 / scale), scale)

    # Evaluate the local expansion at random targets near the origin
    targets = torch.randn(5, 3, dtype=op.dtype, device=op.device) * 0.1
    V_fmm = op.l2p(L_packed, targets, center, scale)

    # Direct reference for the *pure* 1/r kernel
    diff = targets - src_remote
    dists = torch.norm(diff, dim=1)
    V_ref = q / dists

    alpha = (torch.dot(V_fmm, V_ref) / torch.dot(V_ref, V_ref)).item()
    err = torch.norm(V_fmm - V_ref) / torch.norm(V_ref)
    print(f"L2P vs Direct Potential (1/r) | RelErr: {err:.2e}")
    print(f"Global Bias Factor (Target 1.0): {alpha:.4f}")
    if abs(alpha - 1.0) > 0.01:
        print(f"{RED}FAIL{RST} -> Systematic bias detected in L2P or M2L scaling.")
    else:
        print(f"{GRN}PASS{RST} -> Scaling looks correct.")

def test_guards(op):
    print("\n=== TEST 4: Zero Scale Guards ===")
    res = op.p2m(
        torch.zeros(1,3, device=op.device),
        torch.zeros(1, device=op.device),
        torch.zeros(3, device=op.device),
        0.0,
    )
    if torch.all(res == 0):
        print(f"[{GRN}PASS{RST}] P2M handles scale=0.0 gracefully.")
    else:
        print(f"[{RED}FAIL{RST}] P2M returned non-zeros for scale=0.0")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Audit on {device}")
    op = MultipoleOperators(p=8, dtype=torch.float64, device=device)
    try:
        test_rescaling_logic(op)
        test_translation_fixed_scale(op)
        test_l2p_bias(op)
        test_guards(op)
    except Exception as e:
        print(f"\n{RED}CRITICAL ERROR DURING AUDIT:{RST}")
        print(e)
        import traceback
        traceback.print_exc()
