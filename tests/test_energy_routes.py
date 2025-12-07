import json
import os
import math
import tempfile
import subprocess
import sys

PLANE = {
    "domain": "R3",
    "conductors": [
        {"type": "plane", "z": 0.0, "potential": 0.0},
    ],
    "charges": [
        {"type": "point", "q": 1.0e-9, "pos": [0.0, 0.0, 0.25]},
    ],
    "queries": ["potential"],
}

SPHERE_OUT = {
    "domain": "R3",
    "conductors": [
        {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0, "potential": 0.0},
    ],
    "charges": [
        {"type": "point", "q": 1.0e-9, "pos": [0.0, 0.0, 2.0]},
    ],
    "queries": ["potential"],
}

SPHERE_IN = {
    "domain": "R3",
    "conductors": [
        {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0, "potential": 0.0},
    ],
    "charges": [
        {"type": "point", "q": 1.0e-9, "pos": [0.0, 0.0, 0.3]},
    ],
    "queries": ["potential"],
}


def _run_spec(spec):
    """
    Run the CLI on a temporary JSON spec and return (metrics, stdout).

    This uses the `electrodrive.cli` `solve` entrypoint in BEM + cert mode,
    and reads back `metrics.json` from the specified output directory.
    """
    with tempfile.TemporaryDirectory() as td:
        problem_path = os.path.join(td, "spec.json")
        with open(problem_path, "w") as f:
            json.dump(spec, f)

        out_dir = os.path.join(td, "out")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "electrodrive.cli",
            "solve",
            "--problem",
            problem_path,
            "--mode",
            "bem",
            "--cert",
            "--fast",
            "--cert-fast",
            "--out",
            out_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # If the solve fails, surface the CLI’s stderr/stdout in the assertion.
        assert proc.returncode == 0, proc.stderr or proc.stdout

        # Parse metrics.json written by the CLI.
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "r") as mf:
            payload = json.load(mf)

        metrics = payload["metrics"]
        return metrics, proc.stdout


def test_plane_energy_surface_route():
    """
    Grounded plane with a point charge above it.

    - Route B should be a surface-based energy using sigma and the free-space
      potential (the "surface_minus_half_sigma_phi_free" family).
    - Route A and Route B energies must agree to within 1e-3 relative error.
    - For the finite patch representation of the plane, the solver should
      record an effective patch length scale `patch_L` in the metrics.
    """
    m, txt = _run_spec(PLANE)

    # Route B: surface-based energy via sigma and free-space potential.
    assert m["route_B_method"].startswith("surface_minus_half_sigma_phi_free")

    # Energies from Route A and B must agree to within tolerance.
    assert math.isfinite(m["energy_rel_diff"])
    assert m["energy_rel_diff"] <= 1e-3

    # The finite plane patch extent should be recorded in the metrics, not
    # just in logs or stdout. `cli.run_solve` propagates `mesh_stats["patch_L"]`
    # into `metrics["patch_L"]` on a successful BEM solve.
    assert "patch_L" in m, "BEM plane solve should record patch_L in metrics."

    patch_L = m["patch_L"]
    # Require a numeric, strictly positive length scale for the plane patch.
    assert isinstance(
        patch_L, (int, float)
    ), f"patch_L must be numeric, got {type(patch_L)!r}"
    assert patch_L > 0.0, f"patch_L must be positive, got {patch_L!r}"


def test_sphere_outside_analytic():
    """
    Point charge outside a grounded sphere.

    For this configuration, Route B can be evaluated via an analytic
    image-charge solution, so `route_B_method` should reflect that and
    energies must agree to within 1e-3 relative error.
    """
    m, _ = _run_spec(SPHERE_OUT)

    assert m["route_B_method"] == "analytic_sphere_external"
    assert math.isfinite(m["energy_rel_diff"])
    assert m["energy_rel_diff"] <= 1e-3


def test_sphere_inside_surface():
    """
    Point charge inside a grounded sphere.

    In this regime, Route B falls back to a surface-based energy using sigma
    and the free-space potential (same family as the plane case), and the
    two energy routes must agree to within 1e-3 relative error.
    """
    m, _ = _run_spec(SPHERE_IN)

    assert m["route_B_method"].startswith("surface_minus_half_sigma_phi_free")
    assert math.isfinite(m["energy_rel_diff"])
    assert m["energy_rel_diff"] <= 1e-3
