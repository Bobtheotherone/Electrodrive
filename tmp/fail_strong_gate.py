from electrodrive.core.certify import green_badge_decision
from electrodrive.utils.config import EPS_RECIPROCITY, EPS_MAX_PRINCIPLE

m = {
    "bc_residual_linf": 1e-9,
    "pde_residual_linf": 1e-9,
    "energy_rel_diff": float("nan"),
    "dual_route_l2_boundary": float("nan"),
    "max_principle_margin": EPS_MAX_PRINCIPLE * 10.0,  # too large
    "reciprocity_dev": EPS_RECIPROCITY * 10.0,         # too large
}
print("expect False:", green_badge_decision(m, strong=True))
