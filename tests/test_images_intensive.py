import argparse
import json
import os
from pathlib import Path

import torch

from electrodrive.tools import images_discover as cli
from electrodrive.images import search
from electrodrive.images.basis import PointChargeBasis
from electrodrive.images.search import ImageSystem
from electrodrive.orchestration.parser import CanonicalSpec


class DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def test_run_discover_intensive_overrides_defaults(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"domain": "R3", "BCs": "dirichlet", "conductors": [], "charges": []}), encoding="utf-8")

    called = {}

    def fake_discover_images(**kwargs):
        called.update(
            {
                "n_points_override": kwargs.get("n_points_override"),
                "ratio_boundary_override": kwargs.get("ratio_boundary_override"),
                "adaptive_collocation_rounds": kwargs.get("adaptive_collocation_rounds"),
                "restarts": kwargs.get("restarts"),
            }
        )
        return ImageSystem([], torch.zeros(0))

    monkeypatch.setattr(cli, "discover_images", fake_discover_images)
    monkeypatch.setattr(cli, "save_image_system", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        spec=str(spec_path),
        basis="point",
        nmax=2,
        reg_l1=1e-3,
        restarts=None,
        out=str(tmp_path / "out"),
        basis_generator="none",
        basis_generator_mode="static_only",
        model_checkpoint=None,
        solver=None,
        operator_mode=None,
        adaptive_collocation_rounds=None,
        lambda_group=0.0,
        n_points=None,
        ratio_boundary=None,
        aug_boundary=False,
        subtract_physical=False,
        geo_encoder="egnn",
        intensive=True,
    )

    code = cli.run_discover(args)
    assert code == 0
    assert called["n_points_override"] == 8192
    assert called["ratio_boundary_override"] == 0.7
    assert called["adaptive_collocation_rounds"] == 2
    assert called["restarts"] == 3


def test_intensive_candidate_multiplier_and_oom_fallback(monkeypatch):
    os.environ.pop("EDE_IMAGES_INTENSIVE", None)
    calls = {"n_candidates": [], "last_len": None}

    def fake_generate_candidate_basis(spec, basis_types, n_candidates, device, dtype, rng):
        calls["n_candidates"].append(n_candidates)
        elems = []
        for i in range(n_candidates):
            pos = torch.tensor([0.0, 0.0, float(i)], device=device, dtype=dtype)
            elems.append(PointChargeBasis({"position": pos}))
        return elems

    def fake_get_collocation_data(*args, **kwargs):
        pts = torch.zeros(4, 3)
        target = torch.zeros(4)
        mask = torch.zeros(4, dtype=torch.bool)
        return pts, target, mask

    oom_state = {"raised": False}

    def fake_assemble_basis(candidates, colloc_pts, use_operator, device, dtype):
        if not oom_state["raised"]:
            oom_state["raised"] = True
            raise RuntimeError("CUDA out of memory")
        calls["last_len"] = len(candidates)
        return torch.zeros(colloc_pts.shape[0], len(candidates), device=device, dtype=dtype)

    def fake_solve_sparse(A, X, V, is_boundary, logger, **kwargs):
        K = A.shape[1] if isinstance(A, torch.Tensor) else 0
        support_k = kwargs.get("support_k")
        support = list(range(min(K, support_k if support_k is not None else K)))
        return torch.zeros(K), support

    monkeypatch.setattr(search, "generate_candidate_basis", fake_generate_candidate_basis)
    monkeypatch.setattr(search, "get_collocation_data", fake_get_collocation_data)
    monkeypatch.setattr(search, "assemble_basis", fake_assemble_basis)
    monkeypatch.setattr(search, "solve_sparse", fake_solve_sparse)

    spec = CanonicalSpec.from_json({"domain": "R3", "BCs": "dirichlet", "conductors": [], "charges": []})
    logger = DummyLogger()
    system = search.discover_images(
        spec=spec,
        basis_types=["point"],
        n_max=2,
        reg_l1=1e-3,
        restarts=0,
        logger=logger,
        operator_mode=False,
        adaptive_collocation_rounds=1,
        n_points_override=8,
        ratio_boundary_override=0.5,
        intensive=True,
    )

    assert calls["n_candidates"][0] == 32  # 2 * 16 multiplier
    assert calls["last_len"] == 16  # halved after OOM
    assert isinstance(system, ImageSystem)
