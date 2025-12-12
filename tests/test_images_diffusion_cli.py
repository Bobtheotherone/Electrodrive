import argparse
import json
from pathlib import Path

import torch

from electrodrive.tools import images_discover as cli
from electrodrive.images.diffusion_generator import DiffusionBasisGenerator


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _spec_path(tmp_path: Path) -> Path:
    data = {"conductors": [], "charges": [], "domain": {"bbox": [[-1, -1, -1], [1, 1, 1]]}, "BCs": "dirichlet"}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_diffusion_checkpoint_missing_weights_hard_error(monkeypatch, tmp_path: Path):
    spec_path = _spec_path(tmp_path)
    args = argparse.Namespace(
        spec=str(spec_path),
        basis="point",
        nmax=1,
        reg_l1=1e-3,
        restarts=0,
        out=str(tmp_path / "out"),
        basis_generator="diffusion",
        basis_generator_mode="diffusion",
        model_checkpoint=str(tmp_path / "ckpt.pt"),  # nonexistent / empty
        solver=None,
        operator_mode=None,
        adaptive_collocation_rounds=1,
        lambda_group=0.0,
        n_points=None,
        ratio_boundary=None,
        aug_boundary=False,
        subtract_physical=False,
        geo_encoder="egnn",
        intensive=False,
        intensive_flag=False,
    )

    def fake_load_checkpoint(*a, **k):
        return None

    monkeypatch.setattr(cli, "load_lista_from_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(cli, "load_geo_components_from_checkpoint", lambda *a, **k: (None, None))
    code = cli.run_discover(args)
    assert code == 1


def test_diffusion_no_checkpoint_creates_random_generator(monkeypatch, tmp_path: Path, capsys):
    spec_path = _spec_path(tmp_path)
    args = argparse.Namespace(
        spec=str(spec_path),
        basis="point",
        nmax=1,
        reg_l1=1e-3,
        restarts=0,
        out=str(tmp_path / "out2"),
        basis_generator="diffusion",
        basis_generator_mode="diffusion",
        model_checkpoint=None,
        solver=None,
        operator_mode=None,
        adaptive_collocation_rounds=1,
        lambda_group=0.0,
        n_points=None,
        ratio_boundary=None,
        aug_boundary=False,
        subtract_physical=False,
        geo_encoder="egnn",
        intensive=False,
        intensive_flag=False,
    )

    created = {}

    def fake_discover_images(**kwargs):
        created["basis_generator"] = kwargs.get("basis_generator")
        from electrodrive.images.search import ImageSystem

        return ImageSystem([], torch.zeros(0))

    monkeypatch.setattr(cli, "discover_images", fake_discover_images)
    monkeypatch.setattr(cli, "save_image_system", lambda *a, **k: None)
    code = cli.run_discover(args)
    assert code == 0
    assert isinstance(created.get("basis_generator"), DiffusionBasisGenerator)
