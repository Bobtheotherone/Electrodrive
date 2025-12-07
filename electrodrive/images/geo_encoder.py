from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from electrodrive.images.basis import ChargeNodeInfo, CondNodeInfo
from electrodrive.orchestration.parser import CanonicalSpec


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _to_tensor3(value: Sequence[float] | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 3-vector-like object to a tensor on the requested device/dtype."""
    t = torch.as_tensor(value, device=device, dtype=dtype).view(-1)
    if t.numel() < 3:
        t = F.pad(t, (0, 3 - t.numel()))
    return t[:3]


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _conductor_center(cond: dict, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = cond.get("type")
    if t == "plane":
        z = _safe_float(cond.get("z", 0.0))
        return torch.tensor([0.0, 0.0, z], device=device, dtype=dtype)
    if "center" in cond:
        return _to_tensor3(cond.get("center", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
    return torch.zeros(3, device=device, dtype=dtype)


def _conductor_type_id(t: str | None) -> int:
    if t == "plane":
        return 0
    if t == "sphere":
        return 1
    if t == "cylinder":
        return 2
    if t in ("torus", "toroid"):
        return 3
    # Fallback keeps the embedding table bounded.
    return 3


@dataclass
class GeoGraph:
    """Lightweight container for graph-structured geometry."""

    node_pos: torch.Tensor          # [N, 3]
    node_scalar: torch.Tensor       # [N, F_scalar]
    type_ids: torch.Tensor          # [N]
    edge_index: torch.Tensor        # [2, E]
    conductor_meta: List[dict]      # length = n_conductors
    charge_meta: List[dict]         # length = n_charges
    dropped_conductors: List[int]


def build_geo_graph_from_spec(
    spec: CanonicalSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_conductors: int = 32,
    conductor_edge_scale: float = 4.0,
) -> GeoGraph:
    """
    Deterministically map a CanonicalSpec into a graph suitable for the GeoEncoder.

    Nodes are ordered as [conductors..., charges...] so that indices are stable.
    Charge–conductor edges are fully connected; conductor–conductor edges are
    added when centres are closer than `conductor_edge_scale * max(r_i, r_j)`.
    """
    conductor_meta: List[dict] = []
    charge_meta: List[dict] = []

    # Build conductor nodes in spec order.
    for idx, cond in enumerate(getattr(spec, "conductors", [])):
        t = cond.get("type")
        type_id = _conductor_type_id(t)
        center = _conductor_center(cond, device=device, dtype=dtype)
        radius = _safe_float(cond.get("radius", cond.get("minor_radius", 0.0)), default=0.0)
        major_radius = _safe_float(cond.get("major_radius", cond.get("radius", 0.0)), default=0.0)
        minor_radius = _safe_float(cond.get("minor_radius", 0.0), default=0.0)
        potential = _safe_float(cond.get("potential", 0.0), default=0.0)
        conductor_meta.append(
            {
                "index": idx,
                "type": t,
                "type_id": type_id,
                "center": center,
                "radius": radius,
                "major_radius": major_radius,
                "minor_radius": minor_radius,
                "potential": potential,
            }
        )

    dropped: List[int] = []
    if len(conductor_meta) > max_conductors:
        # Keep the most central conductors (smallest ||center||) to stay deterministic.
        order = sorted(
            range(len(conductor_meta)),
            key=lambda i: float(torch.linalg.norm(conductor_meta[i]["center"]).item()),
        )
        keep = set(order[:max_conductors])
        dropped = [conductor_meta[i]["index"] for i in range(len(conductor_meta)) if i not in keep]
        conductor_meta = [conductor_meta[i] for i in range(len(conductor_meta)) if i in keep]
        warnings.warn(
            f"GeoEncoder truncated conductors: kept {len(conductor_meta)} of {len(order)} (max_conductors={max_conductors}).",
            stacklevel=2,
        )

    # Charges follow conductors to keep indices consistent.
    for ch_idx, ch in enumerate(getattr(spec, "charges", [])):
        if ch.get("type") != "point":
            continue
        pos_raw = ch.get("pos", None)
        if pos_raw is None:
            continue
        pos = _to_tensor3(pos_raw, device=device, dtype=dtype)
        q_val = _safe_float(ch.get("q", 0.0), default=0.0)
        charge_meta.append({"index": ch_idx, "pos": pos, "q": q_val})

    n_cond = len(conductor_meta)
    n_charge = len(charge_meta)

    if n_cond + n_charge == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        edge_index = torch.empty(2, 0, device=device, dtype=torch.long)
        return GeoGraph(
            node_pos=empty.view(0, 3),
            node_scalar=empty.view(0, 5),
            type_ids=torch.empty(0, device=device, dtype=torch.long),
            edge_index=edge_index,
            conductor_meta=[],
            charge_meta=[],
            dropped_conductors=dropped,
        )

    node_pos: List[torch.Tensor] = []
    node_scalar: List[torch.Tensor] = []
    type_ids: List[int] = []

    for cond in conductor_meta:
        node_pos.append(cond["center"])
        scalar_feats = torch.tensor(
            [
                cond["radius"],
                cond["major_radius"],
                cond["minor_radius"],
                cond["potential"],
                0.0,  # charge placeholder
            ],
            device=device,
            dtype=dtype,
        )
        node_scalar.append(scalar_feats)
        type_ids.append(cond["type_id"])

    for ch in charge_meta:
        node_pos.append(ch["pos"])
        scalar_feats = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, ch["q"]],
            device=device,
            dtype=dtype,
        )
        node_scalar.append(scalar_feats)
        type_ids.append(4)  # point charge

    edges: List[Tuple[int, int]] = []
    # Charge <-> conductor fully connected (bidirectional).
    for j in range(n_charge):
        cj = n_cond + j
        for i in range(n_cond):
            edges.append((cj, i))
            edges.append((i, cj))

    # Conductor <-> conductor edges based on proximity.
    for i in range(n_cond):
        ci = conductor_meta[i]["center"]
        ri = max(conductor_meta[i]["radius"], conductor_meta[i]["minor_radius"], conductor_meta[i]["major_radius"], 0.0)
        for j in range(i + 1, n_cond):
            cj = conductor_meta[j]["center"]
            rj = max(conductor_meta[j]["radius"], conductor_meta[j]["minor_radius"], conductor_meta[j]["major_radius"], 0.0)
            dist = torch.linalg.norm(ci - cj).item()
            r_max = max(ri, rj, 1e-6)
            if dist <= conductor_edge_scale * r_max:
                edges.append((i, j))
                edges.append((j, i))

    if edges:
        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, device=device, dtype=torch.long)

    return GeoGraph(
        node_pos=torch.stack(node_pos, dim=0),
        node_scalar=torch.stack(node_scalar, dim=0),
        type_ids=torch.tensor(type_ids, device=device, dtype=torch.long),
        edge_index=edge_index,
        conductor_meta=conductor_meta,
        charge_meta=charge_meta,
        dropped_conductors=dropped,
    )


class EGNNLayer(nn.Module):
    """Minimal E(n)-equivariant update block with learned message and coord steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = _mlp(2 * hidden_dim + 1, hidden_dim, hidden_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = _mlp(hidden_dim + hidden_dim, hidden_dim, hidden_dim)

    @staticmethod
    def _scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        if src.numel() == 0:
            return torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
        out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
        return out.index_add(0, index, src)

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            m_aggr = torch.zeros_like(h)
            h_out = h + self.node_mlp(torch.cat([h, m_aggr], dim=-1))
            return x, h_out

        src, dst = edge_index
        rel = x[src] - x[dst]
        d2 = torch.sum(rel * rel, dim=-1, keepdim=True)

        m_ij = self.edge_mlp(torch.cat([h[src], h[dst], d2], dim=-1))
        m_aggr = self._scatter(m_ij, src, h.shape[0])

        coord_coef = torch.tanh(self.coord_mlp(m_ij))
        delta = rel * coord_coef
        delta_x = self._scatter(delta, src, x.shape[0])

        x_out = x + delta_x
        h_out = h + self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        return x_out, h_out


class GeoEncoder(nn.Module):
    """
    Geometry-aware encoder using a lightweight EGNN backbone.

    The encoder consumes graphs from :func:`build_geo_graph_from_spec` and emits
    (z_global, z_nodes) along with ChargeNodeInfo / CondNodeInfo for downstream
    candidate generators.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        n_layers: int = 4,
        type_emb_dim: int = 8,
        scalar_dim: int = 5,
        max_conductors: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.scalar_dim = scalar_dim
        self.type_emb_dim = type_emb_dim
        self.max_conductors = max_conductors

        self.type_embedding = nn.Embedding(5, type_emb_dim)
        self.input_proj = nn.Linear(scalar_dim + type_emb_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_pos: torch.Tensor,
        node_scalar: torch.Tensor,
        type_ids: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the EGNN stack and return pooled + per-node embeddings."""
        if node_pos.numel() == 0:
            z_nodes = torch.zeros(0, self.hidden_dim, device=node_pos.device, dtype=node_pos.dtype)
            return z_nodes.new_zeros(self.hidden_dim), z_nodes

        type_ids = type_ids.to(device=node_pos.device, dtype=torch.long)
        type_ids = torch.clamp(type_ids, min=0, max=self.type_embedding.num_embeddings - 1)
        type_feat = self.type_embedding(type_ids)

        feats = torch.cat([node_scalar, type_feat], dim=-1)
        h = self.input_proj(feats)
        x = node_pos

        for layer in self.layers:
            x, h = layer(x, h, edge_index)

        z_nodes = self.readout(h)
        z_global = z_nodes.mean(dim=0)
        return z_global, z_nodes

    def encode(
        self,
        spec: CanonicalSpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[ChargeNodeInfo], List[CondNodeInfo]]:
        """
        Canonical entry point matching SimpleGeoEncoder's signature.

        Returns
        -------
        z_global : torch.Tensor
            Global pooled embedding [H]
        charge_nodes : List[ChargeNodeInfo]
            Per-charge embeddings + positions
        cond_nodes : List[CondNodeInfo]
            Per-conductor embeddings + centres/type metadata
        """
        self.to(device=device, dtype=dtype)

        graph = build_geo_graph_from_spec(
            spec,
            device=device,
            dtype=dtype,
            max_conductors=self.max_conductors,
        )

        z_global, z_nodes = self.forward(
            graph.node_pos,
            graph.node_scalar,
            graph.type_ids,
            graph.edge_index,
        )

        cond_nodes: List[CondNodeInfo] = []
        for idx, meta in enumerate(graph.conductor_meta):
            cond_nodes.append(
                CondNodeInfo(
                    center=meta["center"],
                    embedding=z_nodes[idx],
                    type_id=meta["type_id"],
                    radius=meta["radius"] if math.isfinite(meta["radius"]) else None,
                )
            )

        charge_nodes: List[ChargeNodeInfo] = []
        offset = len(cond_nodes)
        for j, meta in enumerate(graph.charge_meta):
            charge_nodes.append(
                ChargeNodeInfo(
                    pos=meta["pos"],
                    embedding=z_nodes[offset + j],
                    q=meta["q"],
                )
            )

        return z_global.to(dtype=dtype), charge_nodes, cond_nodes


def _materialize_geo_encoder(obj: Any) -> Optional[nn.Module]:
    if isinstance(obj, GeoEncoder):
        return obj
    if isinstance(obj, nn.Module) and hasattr(obj, "encode"):
        return obj
    if isinstance(obj, Mapping):
        enc = GeoEncoder()
        try:
            enc.load_state_dict(obj, strict=False)
            return enc
        except Exception:
            return None
    return None


def _materialize_basis_generator(obj: Any) -> Optional[nn.Module]:
    try:
        from electrodrive.images.learned_generator import MLPBasisGenerator  # local import to avoid cycles
    except Exception:
        MLPBasisGenerator = None  # type: ignore[assignment]

    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, Mapping) and MLPBasisGenerator is not None:
        gen = MLPBasisGenerator()
        try:
            gen.load_state_dict(obj, strict=False)
            return gen
        except Exception:
            return None
    return None


def _maybe_to_device_dtype(module: Optional[nn.Module], device: torch.device | str | None, dtype: torch.dtype | None) -> Optional[nn.Module]:
    if module is None:
        return None
    try:
        module = module.to(device=device if device is not None else None, dtype=dtype)  # type: ignore[assignment]
    except Exception:
        try:
            module = module.to(device=device if device is not None else None)  # type: ignore[assignment]
        except Exception:
            pass
    return module


def load_geo_components_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    Load GeoEncoder / BasisGenerator components from a checkpoint payload.

    Returns (geo_encoder, basis_generator) where each entry may be None.
    """
    path = Path(checkpoint)
    map_location = device if device is not None else "cpu"
    try:
        payload = torch.load(path, map_location=map_location)
    except Exception:
        return None, None

    geo_encoder = None
    basis_generator = None

    if isinstance(payload, dict):
        geo_encoder = _materialize_geo_encoder(payload.get("geo_encoder"))
        basis_generator = _materialize_basis_generator(payload.get("basis_generator"))
    elif isinstance(payload, GeoEncoder):
        geo_encoder = payload
    elif isinstance(payload, nn.Module) and hasattr(payload, "encode"):
        geo_encoder = payload

    geo_encoder = _maybe_to_device_dtype(geo_encoder, device, dtype)
    basis_generator = _maybe_to_device_dtype(basis_generator, device, dtype)

    try:
        if geo_encoder is not None:
            geo_encoder.eval()
    except Exception:
        pass
    try:
        if basis_generator is not None:
            basis_generator.eval()
    except Exception:
        pass

    return geo_encoder, basis_generator


__all__ = [
    "GeoEncoder",
    "EGNNLayer",
    "GeoGraph",
    "build_geo_graph_from_spec",
    "load_geo_components_from_checkpoint",
]
