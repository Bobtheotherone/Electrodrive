import torch

from electrodrive.experiments.layered_complex_candidates import (
    LayeredComplexBoostConfig,
    build_layered_complex_candidates,
)
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.utils.device import ensure_cuda_available_or_skip


def _layered_spec() -> CanonicalSpec:
    spec = {
        "domain": {"bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]},
        "conductors": [],
        "dielectrics": [
            {"name": "region1", "epsilon": 1.0, "z_min": 0.0, "z_max": 5.0},
            {"name": "slab", "epsilon": 4.0, "z_min": -0.3, "z_max": 0.0},
            {"name": "region3", "epsilon": 1.0, "z_min": -5.0, "z_max": -0.3},
        ],
        "charges": [{"type": "point", "q": 1.0, "pos": [0.0, 0.0, 0.2]}],
        "BCs": "dielectric_interfaces",
        "symmetry": ["rot_z"],
        "queries": [],
    }
    return CanonicalSpec.from_json(spec)


def test_layered_complex_candidates_determinism():
    ensure_cuda_available_or_skip("layered complex candidates determinism")
    device = torch.device("cuda")
    spec = _layered_spec()
    cfg = LayeredComplexBoostConfig(
        enabled=True,
        programs=3,
        blocks_min=2,
        blocks_max=2,
        poles_min=2,
        poles_max=2,
        branches_min=1,
        branches_max=1,
        complex_terms_min=2,
        complex_terms_max=2,
        anchor_terms=1,
        xy_jitter=0.02,
        imag_min_scale=1e-3,
        imag_max_scale=4.0,
        log_cluster_std=0.2,
    )
    cands_a = build_layered_complex_candidates(
        spec,
        device=device,
        dtype=torch.float32,
        seed=123,
        config=cfg,
        max_terms=16,
        domain_scale=1.0,
        exclusion_radius=5e-2,
        allow_real_primitives=True,
    )
    cands_b = build_layered_complex_candidates(
        spec,
        device=device,
        dtype=torch.float32,
        seed=123,
        config=cfg,
        max_terms=16,
        domain_scale=1.0,
        exclusion_radius=5e-2,
        allow_real_primitives=True,
    )
    assert len(cands_a) == len(cands_b)
    for cand_a, cand_b in zip(cands_a, cands_b):
        assert cand_a.program == cand_b.program
        assert len(cand_a.elements) == len(cand_b.elements)
        for elem_a, elem_b in zip(cand_a.elements, cand_b.elements):
            pos_a = elem_a.params.get("position")
            pos_b = elem_b.params.get("position")
            if torch.is_tensor(pos_a) and torch.is_tensor(pos_b):
                assert torch.allclose(pos_a, pos_b)
                assert pos_a.is_cuda
            z_a = elem_a.params.get("z_imag")
            z_b = elem_b.params.get("z_imag")
            if torch.is_tensor(z_a) and torch.is_tensor(z_b):
                assert torch.allclose(z_a, z_b)
