import math
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.images import (
    potential_plane_halfspace,
    potential_sphere_grounded,
    force_on_charge_near_grounded_sphere
)
from electrodrive.core.certify import bc_residual_on_boundary, pde_residual_symbolic
from electrodrive.utils.config import EPS_BC

def spec_plane():
    return CanonicalSpec(
        domain="R3",
        conductors=[{"type":"plane","z":0.0,"potential":0.0}],
        dielectrics=[], charges=[{"type":"point","q":1.0,"pos":[0.0,0.0,0.5]}],
        BCs="Dirichlet", symmetry=["axial"], queries=["potential","work_to_infinity"]
    )

def spec_sphere_inside():
    return CanonicalSpec(
        domain="R3",
        conductors=[{"type":"sphere","center":[0,0,0],"radius":1.0,"potential":0.0}],
        dielectrics=[], charges=[{"type":"point","q":1.0,"pos":[0.3,0.0,0.0]}],
        BCs="Dirichlet", symmetry=["axial"], queries=["potential","force_on_charge"]
    )

def test_plane_bc_and_pde_residual():
    # FIX: Updated call signature (no logger)
    sol = potential_plane_halfspace(1.0,(0.0,0.0,0.5))
    spec = spec_plane()
    bc = bc_residual_on_boundary(sol, spec, n_samples=200)
    assert bc <= EPS_BC
    pde = pde_residual_symbolic(sol, spec, logger=None, n_samples=80)
    assert pde < 1e-4

def test_sphere_force_inside():
    Fx,Fy,Fz = force_on_charge_near_grounded_sphere(1.0,(0.3,0.0,0.0),(0.0,0.0,0.0),1.0)
    assert math.isfinite(Fx+Fy+Fz)
    assert Fx > 0.0






