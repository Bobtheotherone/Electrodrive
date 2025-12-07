from __future__ import annotations

import numpy as np
import torch

from electrodrive.images.search import (
    ImageSystem,
    discover_images,
    get_collocation_data,
)
from electrodrive.orchestration.parser import CanonicalSpec


class RecordingLogger:
    def __init__(self) -> None:
        self.records = []
        self.warnings = []
        self.errors = []

    def info(self, msg: str, **kwargs) -> None:  # pragma: no cover - trivial
        self.records.append((msg, kwargs))

    def warning(self, msg: str, **kwargs) -> None:  # pragma: no cover - trivial
        self.warnings.append((msg, kwargs))

    def error(self, msg: str, **kwargs) -> None:  # pragma: no cover - trivial
        self.errors.append((msg, kwargs))


def _plane_point_spec() -> CanonicalSpec:
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


def test_get_collocation_data_rng_reuse(monkeypatch):
    spec = _plane_point_spec()
    logger = RecordingLogger()
    rng = np.random.default_rng(123)
    state = rng.bit_generator.state

    colloc1 = get_collocation_data(
        spec,
        logger,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=32,
        ratio_override=0.5,
    )

    rng.bit_generator.state = state  # reset the same instance
    colloc2 = get_collocation_data(
        spec,
        logger,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        rng=rng,
        n_points_override=32,
        ratio_override=0.5,
    )

    pts1, tgt1, mask1 = colloc1  # type: ignore[misc]
    pts2, tgt2, mask2 = colloc2  # type: ignore[misc]

    assert torch.equal(pts1, pts2)
    assert torch.equal(tgt1, tgt2)
    assert (mask1 == mask2).all()


def test_get_collocation_data_overrides(monkeypatch):
    spec = _plane_point_spec()
    logger = RecordingLogger()

    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "40")
    monkeypatch.setenv("EDE_IMAGES_RATIO_BOUNDARY", "0.75")
    pts_env, tgt_env, mask_env = get_collocation_data(
        spec,
        logger,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
    )  # type: ignore[misc]

    # Environment should steer counts/ratios.
    assert pts_env.shape[0] >= 30  # allow oracle filtering
    frac_env = float(mask_env.sum().item()) / float(mask_env.shape[0])
    assert 0.4 <= frac_env <= 0.9

    pts_override, tgt_override, mask_override = get_collocation_data(
        spec,
        logger,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_is_boundary=True,
        n_points_override=16,
        ratio_override=0.25,
    )  # type: ignore[misc]

    assert pts_override.shape[0] <= 20
    frac_override = float(mask_override.sum().item()) / float(mask_override.shape[0])
    assert 0.0 <= frac_override <= 0.6

    # Overrides should not inherit the previous env-sized tensors.
    assert pts_override.shape[0] != pts_env.shape[0]


def test_discover_images_adaptive_rounds(monkeypatch):
    spec = _plane_point_spec()
    logger = RecordingLogger()

    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "64")
    monkeypatch.setenv("EDE_IMAGES_RATIO_BOUNDARY", "0.5")
    monkeypatch.setenv("EDE_RUN_ID", "adaptive-test")

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=3,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
        adaptive_collocation_rounds=2,
    )

    assert isinstance(system, ImageSystem)
    assert len(system.elements) <= 3
    assert system.weights.numel() == len(system.elements)
    assert any("Adaptive collocation assembled." in msg for msg, _ in logger.records)


def test_discover_images_adaptive_rounds_env_override(monkeypatch):
    spec = _plane_point_spec()
    logger = RecordingLogger()

    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_ROUNDS", "3")
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "16")
    monkeypatch.setenv("EDE_IMAGES_RATIO_BOUNDARY", "0.5")
    monkeypatch.setenv("EDE_RUN_ID", "adaptive-env-test")

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=2,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
    )

    assert isinstance(system, ImageSystem)
    assert len(system.elements) <= 2
    assert system.weights.numel() == len(system.elements)
    matches = [(msg, kw) for msg, kw in logger.records if "Adaptive collocation assembled." in msg]
    assert matches, "Expected adaptive collocation log entry"
    _, payload = matches[-1]
    assert payload.get("n_rounds") == 3


def test_adaptive_rounds_legacy_passes_env(monkeypatch):
    spec = _plane_point_spec()
    logger = RecordingLogger()

    monkeypatch.delenv("EDE_IMAGES_ADAPTIVE_ROUNDS", raising=False)
    monkeypatch.setenv("EDE_IMAGES_ADAPTIVE_PASSES", "2")
    monkeypatch.setenv("EDE_IMAGES_N_POINTS", "32")
    monkeypatch.setenv("EDE_IMAGES_RATIO_BOUNDARY", "0.5")
    monkeypatch.setenv("EDE_RUN_ID", "adaptive-legacy-test")

    system = discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=2,
        reg_l1=1e-5,
        restarts=0,
        logger=logger,
    )

    assert isinstance(system, ImageSystem)
    assert system.weights.numel() == len(system.elements)
    matches = [(msg, kw) for msg, kw in logger.records if "Adaptive collocation assembled." in msg]
    assert matches, "Expected adaptive collocation log entry"
    _, payload = matches[-1]
    assert payload.get("n_rounds") == 3
