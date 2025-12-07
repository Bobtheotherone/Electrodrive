from __future__ import annotations

import json
from pathlib import Path

import torch

from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig
from electrodrive.images.learned_generator import SimpleGeoEncoder
from electrodrive.images.search import discover_images, ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec


def _load_spec(path: str) -> CanonicalSpec:
    data = json.loads(Path(path).read_text())
    return CanonicalSpec.from_json(data)


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def test_diffusion_generator_emits_candidates():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    encoder = SimpleGeoEncoder(latent_dim=8, hidden_dim=16)
    cfg = DiffusionGeneratorConfig(k_max=6, n_steps=4, hidden_dim=32, n_heads=2)
    generator = DiffusionBasisGenerator(cfg=cfg)

    device = torch.device("cpu")
    dtype = torch.float32
    with torch.no_grad():
        z_global, charge_nodes, cond_nodes = encoder.encode(spec, device=device, dtype=dtype)
        elems = generator(
            z_global=z_global,
            charge_nodes=charge_nodes,
            conductor_nodes=cond_nodes,
            n_candidates=4,
        )

    assert len(elems) <= 4
    for elem in elems:
        pos = elem.params.get("position", None)
        assert pos is not None
        assert torch.isfinite(pos).all()
        assert elem.type in cfg.type_names


def test_discover_images_diffusion_mode_runs():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    encoder = SimpleGeoEncoder(latent_dim=8, hidden_dim=16)
    generator = DiffusionBasisGenerator(DiffusionGeneratorConfig(k_max=8, n_steps=4, hidden_dim=32, n_heads=2))
    logger = _DummyLogger()

    system = discover_images(
        spec=spec,
        basis_types=["sphere_kelvin_ladder", "axis_point"],
        n_max=3,
        reg_l1=1e-3,
        restarts=0,
        logger=logger,
        basis_generator=generator,
        basis_generator_mode="diffusion",
        geo_encoder=encoder,
    )
    assert isinstance(system, ImageSystem)
    assert len(system.elements) > 0
