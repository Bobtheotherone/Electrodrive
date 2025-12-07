from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from electrodrive.images.geo_encoder import GeoEncoder, build_geo_graph_from_spec
from electrodrive.orchestration.parser import CanonicalSpec


def _load_spec(path: str) -> CanonicalSpec:
    return CanonicalSpec.from_json(json.loads(Path(path).read_text()))


def test_build_geo_graph_from_spec_basic():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    device = torch.device("cpu")
    dtype = torch.float32

    graph = build_geo_graph_from_spec(spec, device=device, dtype=dtype)

    assert graph.node_pos.shape == (2, 3)
    assert graph.node_scalar.shape[0] == 2
    assert graph.type_ids.tolist() == [1, 4]
    # Bidirectional charge <-> conductor edges.
    assert graph.edge_index.shape[1] == 2
    assert graph.conductor_meta[0]["type_id"] == 1
    assert pytest.approx(graph.conductor_meta[0]["radius"]) == 1.0


def test_geo_encoder_encode_matches_layout():
    spec = _load_spec("specs/sphere_axis_point_external.json")
    device = torch.device("cpu")
    dtype = torch.float32

    encoder = GeoEncoder(hidden_dim=32, n_layers=2, type_emb_dim=4)
    z_global, charge_nodes, cond_nodes = encoder.encode(spec, device=device, dtype=dtype)

    assert z_global.shape == (32,)
    assert z_global.dtype == dtype
    assert len(cond_nodes) == 1
    assert cond_nodes[0].type_id == 1
    assert cond_nodes[0].center.shape == (3,)
    assert len(charge_nodes) == 1
    assert pytest.approx(charge_nodes[0].q) == spec.charges[0]["q"]
    # Ensure node embeddings align with ordering contract.
    assert charge_nodes[0].embedding.shape[-1] == encoder.hidden_dim
    assert cond_nodes[0].embedding.shape[-1] == encoder.hidden_dim
