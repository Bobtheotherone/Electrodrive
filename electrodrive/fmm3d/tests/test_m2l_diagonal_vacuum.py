import torch
import pytest
from electrodrive.fmm3d.multipole_operators import MultipoleOperators

def direct_potential(x_src, q, x_tgt):
    diff = x_tgt.unsqueeze(1) - x_src.unsqueeze(0)
    r = torch.norm(diff, dim=-1)
    return torch.sum(q.unsqueeze(0) / r, dim=1)

def test_m2l_diagonal_vacuum():
    # Setup: Diagonal translation
    # Source Box Center: (0,0,0)
    # Target Box Center: (3,3,3). Distance = sqrt(27) ~ 5.2
    c_src = torch.zeros(3, dtype=torch.float64)
    c_tgt = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
    t_vec = c_tgt - c_src
    
    # Source particle slightly offset
    x_src = torch.tensor([[0.1, -0.2, 0.1]], dtype=torch.float64)
    q = torch.tensor([1.0], dtype=torch.float64)
    
    # Target particle slightly offset
    x_tgt = c_tgt + torch.tensor([[-0.1, 0.1, -0.2]], dtype=torch.float64)
    
    scale = 1.0
    p = 8
    op = MultipoleOperators(p=p, device=torch.device("cpu"))
    
    # 1. P2M
    M = op.p2m(x_src, q, c_src, scale)
    
    # 2. M2L (Correct vector direction T - S)
    L = op.m2l(M, t_vec, scale)
    
    # 3. L2P
    phi_fmm = op.l2p(L, x_tgt, c_tgt, scale)
    
    # 4. Reference
    phi_ref = direct_potential(x_src, q, x_tgt)
    
    rel_err = torch.abs(phi_fmm - phi_ref) / torch.abs(phi_ref)
    
    print(f"\n[DIAGONAL VACUUM] P={p}, Err={rel_err.item():.3e}")
    print(f"Phi_Ref: {phi_ref.item():.5f}, Phi_FMM: {phi_fmm.item():.5f}")
    
    # If math is robust, error should be small (< 1e-4 or better for p=8)
    # If rotation/math is broken, this will likely be > 1e-2.
    assert rel_err < 1e-4