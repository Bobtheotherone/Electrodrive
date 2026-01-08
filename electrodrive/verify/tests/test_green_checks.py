import torch

from electrodrive.utils.config import K_E
from electrodrive.verify.green_checks import (
    BoundarySpec,
    GreenCheckConfig,
    run_green_checks,
)


def free_space_green(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.linalg.norm(x - y, dim=1).clamp_min(1e-12)
    return K_E / r


def test_green_checks_reciprocity_far_field() -> None:
    cfg = GreenCheckConfig(
        n_samples=128,
        dtype=torch.float64,
        device=torch.device("cpu"),
        far_radius=8.0,
    )
    metrics = run_green_checks(free_space_green, cfg)

    assert metrics["reciprocity_rel_max"] < 1e-8
    rel_err = abs(metrics["far_scaled_mean"] / K_E - 1.0)
    assert rel_err < 1e-3
    assert metrics["far_scaled_cv"] < 1e-3


def test_green_checks_boundary_metrics() -> None:
    cfg = GreenCheckConfig(
        n_samples=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    dirichlet = BoundarySpec(kind="plane", bc_type="dirichlet", normal=(0.0, 0.0, 1.0), offset=0.0)
    metrics_dir = run_green_checks(free_space_green, cfg, boundary=dirichlet)
    assert "dirichlet_max_abs" in metrics_dir
    assert metrics_dir["dirichlet_max_abs"] >= 0.0

    neumann = BoundarySpec(kind="plane", bc_type="neumann", normal=(0.0, 0.0, 1.0), offset=0.0)
    metrics_neu = run_green_checks(free_space_green, cfg, boundary=neumann)
    assert "neumann_max_abs" in metrics_neu
    assert metrics_neu["neumann_max_abs"] >= 0.0
