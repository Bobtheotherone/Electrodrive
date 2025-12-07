from __future__ import annotations

import json
from pathlib import Path

import torch

from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.learned_generator import SimpleGeoEncoder, MLPBasisGenerator
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


def test_mlp_generator_outputs_reasonable_candidates():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    encoder = SimpleGeoEncoder(latent_dim=16, hidden_dim=32)
    generator = MLPBasisGenerator(latent_dim=16, hidden_dim=32, noise_dim=4)

    device = torch.device("cpu")
    dtype = torch.float32
    with torch.no_grad():
        z_global, charge_nodes, cond_nodes = encoder.encode(
            spec,
            device=device,
            dtype=dtype,
        )
        elems = generator(
            z_global=z_global,
            charge_nodes=charge_nodes,
            conductor_nodes=cond_nodes,
            n_candidates=8,
        )

    assert len(elems) == 8
    center = torch.tensor(spec.conductors[0]["center"], device=device, dtype=dtype)
    radius = float(spec.conductors[0]["radius"])
    for elem in elems:
        assert isinstance(elem, PointChargeBasis)
        assert elem.type == "learned_point"
        pos = elem.params["position"]
        assert torch.isfinite(pos).all()
        dist = torch.linalg.norm(pos - center).item()
        assert dist < 5.0 * radius


def test_discover_images_with_learned_only_mode():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    encoder = SimpleGeoEncoder(latent_dim=8, hidden_dim=16)
    generator = MLPBasisGenerator(latent_dim=8, hidden_dim=16, noise_dim=2)
    logger = _DummyLogger()

    system_static = discover_images(
        spec=spec,
        basis_types=["sphere_kelvin_ladder", "axis_point"],
        n_max=4,
        reg_l1=1e-3,
        restarts=0,
        logger=logger,
    )
    assert isinstance(system_static, ImageSystem)
    assert len(system_static.elements) > 0
    assert all(elem.type != "learned_point" for elem in system_static.elements)

    system_learned = discover_images(
        spec=spec,
        basis_types=["sphere_kelvin_ladder", "axis_point"],
        n_max=3,
        reg_l1=1e-3,
        restarts=0,
        logger=logger,
        basis_generator=generator,
        basis_generator_mode="learned_only",
        geo_encoder=encoder,
    )

    assert isinstance(system_learned, ImageSystem)
    assert len(system_learned.elements) > 0
    assert all(elem.type == "learned_point" for elem in system_learned.elements)
