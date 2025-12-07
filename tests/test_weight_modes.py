import numpy as np
import torch

from electrodrive.images.weight_modes import (
    compute_weight_modes,
    fit_symbolic_modes,
    predict_weights_from_modes,
)


def test_compute_weight_modes_rank1():
    rng = np.random.default_rng(123)
    u = rng.standard_normal(6)
    v = rng.standard_normal(4)
    W_true = np.outer(u, v)
    noise = 1e-6 * rng.standard_normal(W_true.shape)
    weights = [torch.tensor(col, dtype=torch.float64) for col in W_true.T + noise.T]

    bundle = compute_weight_modes(weights, [0.1, 0.2, 0.3, 0.4], max_rank=3)

    assert bundle.weights.shape == (6, 4)
    assert bundle.S[0] > 0
    # Rank-1 reconstruction should be extremely accurate on a rank-1 matrix.
    err_rank1 = bundle.recon_error["rank1_rel_fro"]
    assert err_rank1 < 1e-5
    # Singular spectrum should decay quickly for rank-1.
    assert bundle.sigma_norm[1] < 1e-3


def test_fit_symbolic_modes_polynomial():
    z = np.linspace(0.0, 1.0, 8)
    mode_curve = z**2 + 2.0 * z + 1.0  # (z + 1)^2
    mode_curves = np.stack([mode_curve], axis=0)

    fits = fit_symbolic_modes(z, mode_curves, max_rank=1, max_poly_degree=4)

    assert len(fits) == 1
    fit = fits[0]
    assert fit.method == "poly"
    assert fit.rel_rmse < 1e-6
    # Expression should contain the quadratic term.
    assert "z^2" in fit.expression


def test_predict_weights_from_modes_recovers_columns():
    z_grid = [0.0, 0.5, 1.0]
    # Two-mode matrix with simple z dependence for exact fits.
    mode0 = np.array(z_grid) + 1.0
    mode1 = 2.0 * np.array(z_grid) - 0.5
    # Build W = U diag(S) V^T with orthonormal U for simplicity.
    U = np.eye(3)
    S = np.array([3.0, 1.0])
    VT = np.vstack([mode0, mode1])
    W = U[:, :2] @ np.diag(S) @ VT
    weights = [torch.tensor(W[:, j], dtype=torch.float32) for j in range(len(z_grid))]

    bundle = compute_weight_modes(weights, z_grid, max_rank=2)
    fits = fit_symbolic_modes(z_grid, bundle.mode_curves, max_rank=2, max_poly_degree=3)

    for z, expected_col in zip(z_grid, weights):
        w_pred = predict_weights_from_modes(z, {"U": bundle.U, "S": bundle.S}, fits, max_rank=2)
        assert w_pred is not None
        rel_err = np.linalg.norm(w_pred - expected_col.numpy()) / max(
            np.linalg.norm(expected_col.numpy()), 1e-8
        )
        assert rel_err < 1e-4


def test_spectral_gap_and_fit_quality_gating():
    from electrodrive.images.weight_modes import spectral_gap_ok, fit_quality_ok, SymbolicFit

    assert not spectral_gap_ok([0.0, 0.0], rank=1)
    assert spectral_gap_ok([1.0, 0.05], rank=1, thresh=0.1)

    bad_fit = SymbolicFit(
        mode=0,
        method="poly",
        expression="0",
        coefficients={"poly_coeffs": [0.0]},
        rmse=1.0,
        mae=1.0,
        max_abs=1.0,
        rel_rmse=0.5,
    )
    good_fit = SymbolicFit(
        mode=0,
        method="poly",
        expression="z",
        coefficients={"poly_coeffs": [1.0]},
        rmse=0.0,
        mae=0.0,
        max_abs=0.0,
        rel_rmse=0.0,
    )
    assert not fit_quality_ok([bad_fit], rel_rmse_tol=0.2)
    assert fit_quality_ok([good_fit], rel_rmse_tol=0.2)
