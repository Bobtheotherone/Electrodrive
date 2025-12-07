from __future__ import annotations

import json
import os
from pathlib import Path
import math

import torch

from electrodrive.images.basis import generate_candidate_basis
from electrodrive.images.learned_solver import LISTALayer, load_lista_from_checkpoint
from electrodrive.images.geo_encoder import load_geo_components_from_checkpoint
from electrodrive.images.learned_generator import MLPBasisGenerator, SimpleGeoEncoder
from electrodrive.tools.images_discover import run_discover
from electrodrive.images.operator import BasisOperator
from electrodrive.images.search import (
    AugLagrangeConfig,
    ImageSystem,
    _build_boundary_row_weights,
    discover_images,
    assemble_basis_matrix,
    solve_l1_augmented_lagrangian,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.config import K_E


class DummyArgs:
    def __init__(self, spec: Path, out: Path):
        self.spec = str(spec)
        self.basis = "point"
        self.nmax = 1
        self.reg_l1 = 1e-5
        self.restarts = 0
        self.out = str(out)
        self.basis_generator = "none"
        self.basis_generator_mode = "static_only"
        self.model_checkpoint = None
        self.solver = None
        self.operator_mode = None
        self.adaptive_collocation_rounds = 1
        self.lambda_group = 0.0
        self.n_points = None
        self.ratio_boundary = None
        self.aug_boundary = False


def test_images_discover_empty_collocation(tmp_path: Path):
    spec_data = {
        "conductors": [{"type": "plane", "z": 0.0}],
        "charges": [],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

    args = DummyArgs(spec_path, tmp_path / "out")
    code = run_discover(args)
    assert code in (0, 1, 2)


def test_checkpoint_loader_roundtrip(tmp_path: Path) -> None:
    lista = LISTALayer(K=4, n_steps=2)
    geo = SimpleGeoEncoder()
    gen = MLPBasisGenerator()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "lista": lista.state_dict(),
            "geo_encoder": geo.state_dict(),
            "basis_generator": gen.state_dict(),
            "config": {"lista_K": 4, "lista_steps": 2},
        },
        ckpt_path,
    )

    lista_loaded = load_lista_from_checkpoint(ckpt_path)
    geo_loaded, gen_loaded = load_geo_components_from_checkpoint(ckpt_path)

    assert lista_loaded is not None
    assert lista_loaded.K == 4
    assert lista_loaded.training is False
    assert geo_loaded is not None and hasattr(geo_loaded, "encode")
    assert gen_loaded is not None


class DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def _build_plane_point_spec() -> CanonicalSpec:
    return CanonicalSpec.from_json(
        {
            "domain": "R3",
            "BCs": "dirichlet",
            "conductors": [
                {
                    "type": "plane",
                    "z": 0.0,
                    "potential": 0.0,
                }
            ],
            "charges": [
                {
                    "type": "point",
                    "q": 1e-9,
                    "pos": [0.1, -0.2, 0.5],
                }
            ],
        }
    )


def test_discover_images_plane_point_boundary_small():
    """End-to-end smoke test for the sparse image discovery pipeline.

    - simple grounded plane + point charge
    - discover_images should return a non-empty ImageSystem
    - the discovered images should produce much smaller potentials on
      the conductor surface than in the interior (loose tolerance).
    """
    spec = _build_plane_point_spec()
    logger = DummyLogger()

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=8,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
    )

    # Sanity: non-empty sparse image system
    assert isinstance(system, ImageSystem)
    assert len(system.elements) > 0
    assert system.weights.numel() == len(system.elements)

    device = system.weights.device
    dtype = system.weights.dtype

    # Sample a small grid of points above the plane (interior) and on the
    # plane (boundary) and compare |V|.
    n_xy = 32
    xs = torch.linspace(-1.0, 1.0, n_xy, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, n_xy, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")

    # Interior slice at z = 0.5 > 0
    z_int = torch.full_like(XX, 0.5)
    pts_int = torch.stack(
        [XX.reshape(-1), YY.reshape(-1), z_int.reshape(-1)],
        dim=1,
    )

    # Boundary slice on the grounded plane z = 0
    z_bc = torch.zeros_like(XX)
    pts_bc = torch.stack(
        [XX.reshape(-1), YY.reshape(-1), z_bc.reshape(-1)],
        dim=1,
    )

    V_int = system.potential(pts_int)
    V_bc = system.potential(pts_bc)

    # Basic sanity: fields should be finite and nontrivial.
    assert torch.isfinite(V_int).all()
    assert torch.isfinite(V_bc).all()

    mean_int = float(V_int.abs().mean().item())
    mean_bc = float(V_bc.abs().mean().item())

    # Require a non-trivial interior field and a boundary potential that
    # is noticeably smaller (very loose tolerance to avoid brittleness).
    assert mean_int > 1e-6
    assert mean_bc < 0.9 * mean_int


def test_discover_images_lista_matches_ista_support() -> None:
    """LISTA solver matches ISTA on a simple plane + point spec."""
    os.environ["EDE_DEVICE"] = "cpu"
    os.environ["EDE_IMAGES_SHUFFLE_CANDIDATES"] = "0"
    spec = _build_plane_point_spec()
    logger = DummyLogger()

    n_max = 4
    reg = 1e-6
    basis_types = ["point"]

    system_ista = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg,
        restarts=0,
        logger=logger,
        solver="ista",
    )

    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=max(1, n_max * 4),
        device=system_ista.weights.device,
        dtype=system_ista.weights.dtype,
    )
    lista = LISTALayer(K=len(candidates), n_steps=6, init_theta=1e-6)

    system_lista = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg,
        restarts=0,
        logger=logger,
        solver="lista",
        lista_model=lista,
    )

    assert len(system_lista.elements) == len(system_ista.elements)
    types_ista = [e.type for e in system_ista.elements]
    types_lista = [e.type for e in system_lista.elements]
    assert types_lista == types_ista

    grid = torch.tensor(
        [
            [0.0, 0.0, 0.2],
            [0.1, -0.1, 0.4],
            [-0.2, 0.3, 0.6],
        ],
        device=system_ista.weights.device,
        dtype=system_ista.weights.dtype,
    )
    V_ista = system_ista.potential(grid)
    V_lista = system_lista.potential(grid)
    rel_err = torch.abs(V_lista - V_ista) / (torch.abs(V_ista) + 1e-6)
    assert torch.max(rel_err).item() < 0.4


def test_dense_operator_matvec_rmatvec_match_plane(monkeypatch) -> None:
    """Dense matrix and BasisOperator agree on matvec/rmatvec for a small plane spec."""
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "0")
    spec = _build_plane_point_spec()
    dtype = torch.float64
    device = torch.device("cpu")

    # Minimal collocation set on and above the plane.
    X = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.1, -0.1, 0.4],
            [0.05, 0.2, 0.6],
            [-0.15, 0.05, 0.8],
        ],
        dtype=dtype,
        device=device,
    )

    candidates = generate_candidate_basis(
        spec,
        basis_types=["point"],
        n_candidates=3,
        device=device,
        dtype=dtype,
    )

    A_dense = assemble_basis_matrix(candidates, X)
    A_op = BasisOperator(candidates, points=X, device=device, dtype=dtype)

    w = torch.tensor([0.3, -0.6, 0.2], dtype=dtype, device=device)
    r = torch.tensor([0.5, -0.25, 0.1, 0.2], dtype=dtype, device=device)

    mat_dense = A_dense @ w
    mat_op = A_op.matvec(w, X)
    assert torch.allclose(mat_op, mat_dense, atol=1e-10, rtol=1e-7)

    rmat_dense = A_dense.T @ r
    rmat_op = A_op.rmatvec(r, X)
    assert torch.allclose(rmat_op, rmat_dense, atol=1e-10, rtol=1e-7)


def _load_spec(path: Path) -> CanonicalSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CanonicalSpec.from_json(raw)


def test_legacy_dense_vs_operator_plane(monkeypatch) -> None:
    """Legacy dense path stays consistent with operator-mode solve."""
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "0")
    monkeypatch.setenv("EDE_RUN_ID", "legacy-plane")
    spec = _build_plane_point_spec()
    logger = DummyLogger()

    system_dense = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=4,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=False,
        adaptive_collocation_rounds=1,
    )

    monkeypatch.setenv("EDE_RUN_ID", "legacy-plane")
    system_op = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=4,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=True,
        adaptive_collocation_rounds=1,
    )

    assert len(system_dense.elements) > 0
    assert len(system_op.elements) > 0

    grid = torch.tensor(
        [
            [0.0, 0.0, 0.2],
            [0.1, -0.1, 0.4],
            [-0.2, 0.3, 0.6],
        ],
        device=system_dense.weights.device,
        dtype=system_dense.weights.dtype,
    )
    v_dense = system_dense.potential(grid)
    v_op = system_op.potential(grid).to(device=grid.device, dtype=grid.dtype)
    rel_err = torch.abs(v_dense - v_op) / (torch.abs(v_dense) + 1e-6)
    assert float(rel_err.max().item()) < 0.55


def test_lista_operator_mode_parity_with_ista(monkeypatch) -> None:
    """LISTA + operator-mode discovery stays close to ISTA on the plane spec."""
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "0")
    spec = _build_plane_point_spec()
    logger = DummyLogger()

    basis_types = ["point"]
    n_max = 4
    reg = 1e-6
    dtype = torch.float32
    device = torch.device("cpu")

    # Build LISTA with matching candidate dimension.
    candidates = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=max(1, n_max * 4),
        device=device,
        dtype=dtype,
    )
    lista = LISTALayer(K=len(candidates), n_steps=6, init_theta=1e-6)

    system_ista = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=True,
    )

    system_lista = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=n_max,
        reg_l1=reg,
        restarts=0,
        logger=logger,
        solver="lista",
        operator_mode=True,
        lista_model=lista,
    )

    assert len(system_lista.elements) > 0

    # Boundary-only sample on z = 0 to measure residuals.
    xy = torch.linspace(-0.6, 0.6, 5, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xy, xy, indexing="xy")
    pts_bc = torch.stack([XX.reshape(-1), YY.reshape(-1), torch.zeros_like(XX).reshape(-1)], dim=1)

    bc_err_ista = float(system_ista.potential(pts_bc).abs().mean().item())
    bc_err_lista = float(system_lista.potential(pts_bc).abs().mean().item())

    # Allow modest degradation but guard against large boundary regressions.
    tol = max(5e-4, bc_err_ista * 2.5 + 5e-3, bc_err_ista + 0.15)
    assert bc_err_lista <= min(tol, 0.4)


def test_legacy_dense_vs_operator_sphere(monkeypatch) -> None:
    """Legacy dense path parity on a small grounded sphere spec."""
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "0")
    monkeypatch.setenv("EDE_RUN_ID", "legacy-sphere")
    spec = _load_spec(Path("specs") / "sphere_axis_point_external.json")
    logger = DummyLogger()

    basis_types = ["axis_point"]

    system_dense = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=3,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=False,
        adaptive_collocation_rounds=1,
    )

    monkeypatch.setenv("EDE_RUN_ID", "legacy-sphere")
    system_op = discover_images(
        spec=spec,
        basis_types=basis_types,
        n_max=3,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=True,
        adaptive_collocation_rounds=1,
    )

    assert len(system_dense.elements) > 0
    assert len(system_op.elements) > 0

    eval_pts = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [0.2, -0.1, 0.8],
            [-0.3, 0.25, 1.2],
        ],
        device=system_dense.weights.device,
        dtype=system_dense.weights.dtype,
    )
    v_dense = system_dense.potential(eval_pts)
    v_op = system_op.potential(eval_pts).to(device=eval_pts.device, dtype=eval_pts.dtype)
    rel_err = torch.abs(v_dense - v_op) / (torch.abs(v_dense) + 1e-6)
    assert float(rel_err.max().item()) < 0.35


def test_stage0_sphere_kelvin_ladder_sanity(monkeypatch) -> None:
    """Kelvin-ladder sphere discovery finds interior images with low boundary error."""
    monkeypatch.setenv("EDE_DEVICE", "cpu")
    monkeypatch.setenv("EDE_IMAGES_SHUFFLE_CANDIDATES", "0")
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "96")
    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "1")
    monkeypatch.setenv("EDE_IMAGES_DTYPE", "float64")

    spec = _load_spec(Path("specs") / "sphere_axis_point_external.json")
    logger = DummyLogger()

    system = discover_images(
        spec=spec,
        basis_types=["sphere_kelvin_ladder"],
        n_max=4,
        reg_l1=1e-4,
        restarts=0,
        logger=logger,
        solver="ista",
        operator_mode=True,
        adaptive_collocation_rounds=1,
    )

    assert len(system.elements) > 0
    center = torch.tensor(spec.conductors[0]["center"], device=system.weights.device, dtype=system.weights.dtype)
    radius = float(spec.conductors[0]["radius"])
    charge_pos = torch.tensor(spec.charges[0]["pos"], device=system.weights.device, dtype=system.weights.dtype)
    charge_q = float(spec.charges[0].get("q", 1.0))

    inside = False
    for elem in system.elements:
        pos = elem.params.get("position", None)
        if pos is None:
            continue
        pos_t = pos.to(device=system.weights.device, dtype=system.weights.dtype)
        dist = torch.linalg.norm(pos_t - center).item()
        if 0.0 < dist < radius:
            inside = True
            break
    assert inside

    # Sample a modest sphere-surface grid to measure boundary residuals.
    theta = torch.linspace(0.0, 2 * math.pi, steps=6, device=center.device, dtype=center.dtype)
    phi = torch.linspace(0.2 * math.pi, 0.8 * math.pi, steps=5, device=center.device, dtype=center.dtype)
    pts = []
    for th in theta:
        for ph in phi:
            x = radius * torch.sin(ph) * torch.cos(th)
            y = radius * torch.sin(ph) * torch.sin(th)
            z = radius * torch.cos(ph)
            pts.append(center + torch.stack([x, y, z], dim=0))
    pts_bc = torch.stack(pts, dim=0)

    V_bc = system.potential(pts_bc)
    bc_mean = float(V_bc.abs().mean().item())
    bc_max = float(V_bc.abs().max().item())

    fs_potential = K_E * charge_q / torch.linalg.norm(pts_bc - charge_pos, dim=1).clamp_min(1e-6)
    fs_scale = float(fs_potential.abs().max().item())
    rel_mean = bc_mean / (fs_scale + 1e-12)
    rel_max = bc_max / (fs_scale + 1e-12)

    assert rel_mean < 0.15
    assert rel_max < 0.25


def test_boundary_weight_ratio_and_mix_modes() -> None:
    mask = torch.tensor([True, False, True, False], dtype=torch.bool)

    rw_ratio, _, mode_ratio, metric_ratio, n_b, n_i = _build_boundary_row_weights(
        mask,
        boundary_weight=3.0,
        boundary_mode="mix",
        boundary_penalty_default=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mode_ratio == "ratio"
    assert metric_ratio == 3.0
    assert n_b == 2 and n_i == 2
    assert torch.allclose(rw_ratio[mask], torch.full((2,), 3.0))
    assert torch.allclose(rw_ratio[~mask], torch.ones(2))

    rw_mix, _, mode_mix, metric_mix, _, _ = _build_boundary_row_weights(
        mask,
        boundary_weight=0.25,
        boundary_mode="mix",
        boundary_penalty_default=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mode_mix == "mix"
    assert metric_mix == 0.25
    assert torch.allclose(rw_mix[mask], torch.full((2,), 0.25))
    assert torch.allclose(rw_mix[~mask], torch.full((2,), 0.75))


def test_augmented_lagrangian_reduces_boundary_norm() -> None:
    logger = DummyLogger()
    A = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    g = torch.tensor([1.0, 0.0], dtype=torch.float32)
    boundary_mask = torch.tensor([False, True], dtype=torch.bool)
    cfg = AugLagrangeConfig(rho0=10.0, rho_growth=10.0, rho_max=1e4, max_outer=2, base_tol=1e-6)

    w, A_last, g_last, row_w, diag = solve_l1_augmented_lagrangian(
        g,
        boundary_mask,
        make_weighted_dict=lambda weights: A * weights.sqrt().view(-1, 1),
        predict_unweighted=lambda w_vec, mask=None: (A if mask is None else A[mask]) @ w_vec,
        collocation=torch.zeros((2, 3), dtype=torch.float32),
        reg_l1=1e-5,
        logger=logger,
        cfg=cfg,
    )

    assert w.shape[0] == 1
    assert row_w.shape[0] == g.shape[0]
    assert diag["n_outer"] == 2
    assert diag["bc_norm_after"] <= diag["bc_norm_before"] + 1e-6
    assert torch.isfinite(w).all()
    assert torch.isfinite(g_last).all()
    assert torch.isfinite(A_last if isinstance(A_last, torch.Tensor) else torch.tensor(0.0)).all()
