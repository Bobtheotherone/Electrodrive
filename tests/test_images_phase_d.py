import torch
from electrodrive.orchestration.parser import CanonicalSpec

from electrodrive.images.basis import PointChargeBasis, annotate_group_info
from electrodrive.images.diffusion_generator import DiffusionBasisGenerator, DiffusionGeneratorConfig
from electrodrive.images.learned_solver import LISTALayer
from electrodrive.images.search import _build_learned_candidates
from electrodrive.images.search import ImageSystem, optimize_parameters_lbfgs


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def test_lista_group_sparsity_zeroes_group_when_lambda_large():
    K = 2
    lista = LISTALayer(K=K, n_steps=5)
    A = torch.eye(K)
    b = torch.tensor([1.0, 0.5])
    group_ids = torch.tensor([0, 0])
    w = lista.solve(A, b, group_ids=group_ids, lambda_group=10.0)
    assert torch.allclose(w, torch.zeros_like(w))


def test_diffusion_candidates_respect_slab_bounds_and_groups(monkeypatch):
    cfg = DiffusionGeneratorConfig(k_max=4, n_steps=1, hidden_dim=8, n_layers=1, n_heads=1)
    gen = DiffusionBasisGenerator(cfg)
    gen.set_slab_bounds((-0.3, 0.0))

    def fake_sample(cond_vec, mask, device, dtype, n_samples):
        x = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, -0.1], [0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]], device=device, dtype=dtype)
        logits = torch.zeros(1, 4, len(gen.type_names), device=device, dtype=dtype)
        mask_out = torch.ones(1, 4, device=device, dtype=torch.bool)
        return x, logits, mask_out

    monkeypatch.setattr(gen, "sample", fake_sample)
    elems = gen(
        z_global=torch.zeros(4),
        charge_nodes=[],
        conductor_nodes=[],
        n_candidates=3,
    )
    assert len(elems) == 3
    conductor_ids = [getattr(e, "_group_info", {}).get("conductor_id") for e in elems]
    families = [getattr(e, "_group_info", {}).get("family_name") for e in elems]
    assert conductor_ids == [0, 1, 2]
    assert all(fam == "three_layer_diffusion" for fam in families)


def test_build_learned_candidates_sets_slab_bounds(monkeypatch):
    class FakeGen:
        def __init__(self):
            self.slab_bounds_set = None

        def set_slab_bounds(self, bounds):
            self.slab_bounds_set = bounds

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return []

    class FakeEncoder:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def encode(self, spec, device, dtype):
            z_global = torch.zeros(4, device=device, dtype=dtype)
            return z_global, [], []

    spec = CanonicalSpec.from_json(
        {
            "BCs": "dielectric_interfaces",
            "dielectrics": [
                {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
                {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
                {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
            ],
            "charges": [],
            "domain": {"bbox": [[-1, -1, -1], [1, 1, 1]]},
        }
    )
    fake_gen = FakeGen()
    fake_enc = FakeEncoder()
    _build_learned_candidates(
        spec,
        basis_generator=fake_gen,
        geo_encoder=fake_enc,
        n_candidates=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=DummyLogger(),
    )
    assert fake_gen.slab_bounds_set == (-0.3, 0.0)


def test_lbfgs_reverts_on_out_of_bounds(monkeypatch):
    logger = DummyLogger()
    elem = PointChargeBasis({"position": torch.tensor([0.0, 0.0, 0.1])})
    annotate_group_info(elem, conductor_id=0, family_name="axis_point", motif_index=0)
    system = ImageSystem([elem], torch.tensor([1.0]))
    points = torch.zeros(4, 3)
    targets = torch.zeros(4)

    class FakeLBFGS:
        def __init__(self, params, **kwargs):
            self.params = list(params)

        def step(self, closure):
            _ = closure()
            for p in self.params:
                p.data = torch.full_like(p, 10.0)
            return torch.tensor(0.0)

        def zero_grad(self, set_to_none=True):
            pass

    monkeypatch.setattr(torch.optim, "LBFGS", FakeLBFGS)
    refined = optimize_parameters_lbfgs(
        system,
        points,
        targets,
        logger,
        domain_extent=1.0,
        max_iter_override=10,
    )
    pos_final = refined.elements[0].params["position"]
    assert torch.allclose(pos_final, torch.tensor([0.0, 0.0, 0.1]))
    assert torch.allclose(refined.weights, torch.tensor([1.0]))
