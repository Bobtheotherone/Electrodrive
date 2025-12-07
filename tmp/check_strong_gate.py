from electrodrive.core.certify import green_badge_decision

m = {
    "bc_residual_linf": 1e-9,
    "pde_residual_linf": 1e-9,
    "energy_rel_diff": float("nan"),          # not computed => not a veto
    "dual_route_l2_boundary": float("nan"),   # not computed => not a veto
    "max_principle_margin": float("nan"),     # not computed => not a veto (strong)
    "reciprocity_dev": float("nan"),          # not computed => not a veto (strong)
}
print("base gate  :", green_badge_decision(m))
print("strong gate:", green_badge_decision(m, strong=True))
