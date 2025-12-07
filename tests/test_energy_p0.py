import math

import pytest
import torch

from electrodrive.core.images import potential_plane_halfspace
from electrodrive.core.certify import energy_consistency_check
from electrodrive.utils.config import K_E


def test_plane_energy_consistency_p0_spec_formula():
    """
    P0.2: Dual-route energy consistency for the plane case, using the
    analytic image-charge solution as the oracle.

    We consider a single point charge q at height d above a grounded plane.
    The analytic plane solution gives the induced potential at the charge,
    and Route A (charge-based) energy should match the expected formula.
    """
    q = 1.0
    d = 1.0
    r0 = (0.0, 0.0, d)

    spec = {
        "domain": "R3",
        "BCs": "Dirichlet",
        "conductors": [
            {"type": "plane", "z": 0.0, "potential": 0.0},
        ],
        "charges": [
            {"type": "point", "q": q, "pos": list(r0)},
        ],
    }

    # Analytic solution (plane image method)
    sol = potential_plane_halfspace(q, r0)

    # Expected Route A (grounded conductor): U = -0.5 * q * phi_induced
    # Plane image: phi_induced(r0) = -K_E * q / (2*d)
    phi_induced_expected = -K_E * q / (2.0 * d)
    expected_W_A = -0.5 * q * phi_induced_expected  # -> +K_E*q^2/(4d)

    metrics = energy_consistency_check(sol, spec, logger=None)

    W_A = metrics["energy_A"]
    W_B = metrics["energy_B"]
    rel_diff = metrics["energy_rel_diff"]

    # Route A should match the analytic formula extremely well.
    assert math.isclose(W_A, expected_W_A, rel_tol=1e-9)
    assert metrics["route_A_method"] == "charge_minus_half_q_phi_induced"

    # For this analytic-only solution, Route B (surface) is not computed.
    assert math.isnan(W_B)
    assert math.isnan(rel_diff)
