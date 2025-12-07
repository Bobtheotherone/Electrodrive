from __future__ import annotations

from electrodrive.core.identities import (
    compute_mesh_hash,
    compute_problem_hash,
    compute_kernel_hash,
)


def test_mesh_hash_permutation_and_rotation_invariance():
    V = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    F = [
        [0, 1, 2],
        [1, 3, 2],
    ]

    h_ref = compute_mesh_hash(V, F)

    V_perm = [V[1], V[0], V[3], V[2]]
    idx_map = {0: 1, 1: 0, 2: 3, 3: 2}
    F_perm = [[idx_map[i] for i in tri] for tri in [F[1], F[0]]]
    h_perm = compute_mesh_hash(V_perm, F_perm)

    F_rot = [[1, 2, 0], [3, 2, 1]]
    h_rot = compute_mesh_hash(V, F_rot)

    assert h_ref == h_perm
    assert h_ref == h_rot


def test_problem_hash_float_and_special_values_stable():
    spec1 = {"a": 1.0, "b": float("nan"), "c": float("inf"), "d": -float("inf")}
    spec2 = {"d": -float("inf"), "c": float("inf"), "b": float("nan"), "a": 1.0}
    h1 = compute_problem_hash(spec1)
    h2 = compute_problem_hash(spec2)
    assert h1 == h2


def test_kernel_hash_whitelist():
    params = {"a": 1, "b": 2, "c": 3}
    h1 = compute_kernel_hash(params, whitelist=["a", "b"])
    h2 = compute_kernel_hash({"b": 2, "a": 1, "x": 9}, whitelist=["b", "a"])
    assert h1 == h2


def test_mesh_hash_accepts_array_like_and_numpy():
    """Mesh hashing must be stable across list and numpy inputs."""
    import numpy as np

    V_list = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    F_list = [[0, 1, 2]]

    # List-of-lists input
    h_list = compute_mesh_hash(V_list, F_list)

    # Explicit numpy arrays (same geometry)
    V_np = np.asarray(V_list, dtype=float)
    F_np = np.asarray(F_list, dtype=int)
    h_np = compute_mesh_hash(V_np, F_np)

    assert h_list == h_np
