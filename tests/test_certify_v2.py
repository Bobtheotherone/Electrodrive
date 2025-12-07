from electrodrive.core.certify import green_badge_decision
from electrodrive.utils.config import EPS_BC, EPS_DUAL, EPS_PDE, EPS_ENERGY

def test_green_badge_pass_minimal():
    m = {
        "bc_residual_linf": EPS_BC*0.1,
        "dual_route_l2_boundary": EPS_DUAL*0.1,
        "pde_residual_linf": EPS_PDE*0.1,
        "energy_rel_diff": EPS_ENERGY*0.1,
        "mean_value_deviation": 1e-8,
    }
    assert green_badge_decision(m) is True

def test_green_badge_fail_bc():
    m = {
        "bc_residual_linf": EPS_BC*10.0,
        "dual_route_l2_boundary": EPS_DUAL*0.1,
        "pde_residual_linf": EPS_PDE*0.1,
        "energy_rel_diff": EPS_ENERGY*0.1,
    }
    assert green_badge_decision(m) is False

def test_green_badge_fail_energy_p0():
    # P0.2: Energy consistency must be met if finite.
    m = {
        "bc_residual_linf": EPS_BC*0.1,
        "dual_route_l2_boundary": EPS_DUAL*0.1,
        "pde_residual_linf": EPS_PDE*0.1,
        "energy_rel_diff": EPS_ENERGY*10.0, # Fails here
    }
    assert green_badge_decision(m) is False

def test_green_badge_pass_energy_nan_p0():
    # P0.2: If energy consistency is NaN (not computed), it should pass.
    m = {
        "bc_residual_linf": EPS_BC*0.1,
        "dual_route_l2_boundary": EPS_DUAL*0.1,
        "pde_residual_linf": EPS_PDE*0.1,
        "energy_rel_diff": float("nan"),
    }
    assert green_badge_decision(m) is True

