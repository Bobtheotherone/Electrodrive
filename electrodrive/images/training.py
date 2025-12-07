from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from electrodrive.images.basis import (
    BasisGenerator,
    ImageBasisElement,
    annotate_group_info,
    generate_candidate_basis,
)
from electrodrive.images.geo_encoder import GeoEncoder
from electrodrive.images.learned_generator import MLPBasisGenerator, SimpleGeoEncoder
from electrodrive.images.learned_solver import LISTALayer
from electrodrive.images.operator import BasisOperator
from electrodrive.images.diffusion_generator import (
    DiffusionBasisGenerator,
    DiffusionGeneratorConfig,
)
from electrodrive.images.search import ImageSystem, get_collocation_data
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.orchestration.spec_registry import (
    load_stage0_sphere_external,
    load_stage1_sphere_dimer_inside,
    list_stage1_variants,
)
from electrodrive.utils.logging import JsonlLogger


# -----------------------------------------------------------------------------#
# Config dataclasses
# -----------------------------------------------------------------------------#


@dataclass
class Stage0Ranges:
    """Sampling ranges for Stage-0 z-positions expressed in radii."""

    plane_z: Tuple[float, float] = (0.35, 3.0)
    sphere_external: Tuple[float, float] = (1.05, 3.0)
    sphere_internal: Tuple[float, float] = (0.2, 0.9)
    q: Tuple[float, float] = (0.5, 2.0)


@dataclass
class Stage1Ranges:
    """Sampling ranges for Stage-1 sphere dimer lens tasks."""

    charge_frac: Tuple[float, float] = (0.1, 0.9)
    gap_margin: float = 0.02
    q: Tuple[float, float] = (0.5, 2.0)


@dataclass
class Stage2Ranges:
    """Placeholder for Stage-2 periodic array curriculum."""

    q: Tuple[float, float] = (0.5, 2.0)
    lateral_jitter: float = 0.0
    note: str = "Periodic gratings (Stage-2) require BEM/FMM periodic kernels."


@dataclass
class DiffusionTrainingConfig:
    """Minimal config scaffold for diffusion BasisGenerator pretraining."""

    out_dir: Path
    k_max: int = 16
    type_names: Tuple[str, ...] = ("learned_point", "axis_point", "point")
    lr: float = 1e-3
    batch_size: int = 4
    n_steps: int = 10
    device: Optional[str] = None
    dtype: str = "float32"
    seed: int = 12345
    lambda_chamfer: float = 1.0
    lambda_type: float = 1.0
    lambda_extra: float = 0.1


@dataclass
class BilevelTrainConfig:
    """Lightweight config for Stage-0 bilevel training with unrolled LISTA."""

    out_dir: Path
    stage: int = 0
    max_steps: int = 200
    batch_size: int = 8
    n_candidates_static: int = 64
    n_candidates_learned: int = 0
    basis_plane: List[str] = field(default_factory=lambda: ["mirror_stack", "point"])
    basis_sphere: List[str] = field(
        default_factory=lambda: ["sphere_kelvin_ladder", "axis_point", "point"]
    )
    basis_dimer: List[str] = field(
        default_factory=lambda: ["sphere_kelvin_ladder", "axis_point", "point"]
    )
    n_points_train: int = 512
    n_points_val: int = 512
    ratio_boundary_train: float = 0.5
    ratio_boundary_val: float = 0.5
    lambda_bc: float = 50.0
    lambda_l1: float = 1e-4
    lambda_group: float = 0.0
    lista_steps: int = 10
    lista_rank: int = 0
    lista_dense_threshold: int = 512
    lr_lista: float = 1e-3
    lr_geo: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: Optional[str] = None
    dtype: str = "float32"
    seed: int = 12345
    checkpoint_every: int = 0
    stage0_geoms: List[str] = field(
        default_factory=lambda: ["plane", "sphere_external"]
    )
    stage1_include_variants: bool = True
    ranges: Stage0Ranges = field(default_factory=Stage0Ranges)
    ranges_stage1: Stage1Ranges = field(default_factory=Stage1Ranges)
    ranges_stage2: Stage2Ranges = field(default_factory=Stage2Ranges)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float64", "double", "fp64"):
        return torch.float64
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32


def _spec_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "specs"


def _load_spec(name: str) -> CanonicalSpec:
    path = _spec_dir() / name
    data = path.read_text(encoding="utf-8")
    return CanonicalSpec.from_json(json.loads(data))


def _load_stage1_specs(include_variants: bool) -> List[CanonicalSpec]:
    specs = [load_stage1_sphere_dimer_inside()]
    if include_variants:
        for info in list_stage1_variants(include_canonical=False):
            try:
                data = info.path.read_text(encoding="utf-8")
                specs.append(CanonicalSpec.from_json(json.loads(data)))
            except Exception:
                continue
    return specs


def _clone_spec_with_z(spec: CanonicalSpec, z0: float, *, q: Optional[float] = None, charge_idx: int = 0) -> CanonicalSpec:
    blob = spec.to_json()
    charges = blob.get("charges")
    if not charges or charge_idx >= len(charges):
        raise ValueError("Spec has no charges to reposition.")
    charges[charge_idx]["pos"][2] = float(z0)
    if q is not None:
        charges[charge_idx]["q"] = float(q)
    blob["charges"] = charges
    return CanonicalSpec.from_json(blob)


def _geom_label(spec: CanonicalSpec) -> str:
    ctypes = sorted({c.get("type") for c in getattr(spec, "conductors", [])})
    if ctypes == ["plane"]:
        return "plane"
    if ctypes == ["sphere"]:
        return "sphere"
    if ctypes.count("sphere") == 2:
        return "sphere_dimer"
    return "generic"


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() > 0 else x.new_tensor(0.0)


def _chamfer_distance(pred: torch.Tensor, target: torch.Tensor, mask_pred: torch.Tensor, mask_tgt: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two masked point sets."""
    if pred.numel() == 0 or target.numel() == 0 or not bool(mask_pred.any()) or not bool(mask_tgt.any()):
        return pred.new_tensor(0.0)
    p = pred[mask_pred]
    q = target[mask_tgt]
    # [P, Q]
    dists = torch.cdist(p, q, p=2)
    if dists.numel() == 0:
        return pred.new_tensor(0.0)
    d1 = torch.min(dists, dim=1).values
    d2 = torch.min(dists, dim=0).values
    return 0.5 * (d1.mean() + d2.mean())


def build_diffusion_training_sample_from_system(
    spec: CanonicalSpec,
    system: "ImageSystem",
    z_global: torch.Tensor,
    z_charge: torch.Tensor,
    *,
    k_max: int,
    type_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Build a padded diffusion training sample from a discovered image system.

    Local frame mapping is currently identity; future work can align to
    symmetry axes (e.g., plane normal or sphere center) before padding.
    """
    X = torch.zeros(k_max, 3, device=device, dtype=dtype)
    type_idx = torch.zeros(k_max, device=device, dtype=torch.long)
    mask = torch.zeros(k_max, device=device, dtype=torch.bool)

    type_to_idx = {name: i for i, name in enumerate(type_names)}
    for j, elem in enumerate(system.elements[:k_max]):
        pos = elem.params.get("position", None)
        if pos is None:
            continue
        X[j] = torch.as_tensor(pos, device=device, dtype=dtype).view(3)
        mask[j] = True
        type_idx[j] = type_to_idx.get(elem.type, len(type_names) - 1)

    sample = {
        "X_target": X,
        "type_target": type_idx,
        "mask": mask,
        "cond_global": z_global.detach(),
        "cond_charge": z_charge.detach(),
    }
    return sample


def diffusion_set_loss(
    pred_X: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_mask: torch.Tensor,
    target: Dict[str, torch.Tensor],
    *,
    lambda_chamfer: float,
    lambda_type: float,
    lambda_extra: float,
) -> torch.Tensor:
    """Set-matching loss combining Chamfer distance and type alignment."""
    tgt_X = target["X_target"]
    tgt_mask = target["mask"]
    cham = _chamfer_distance(pred_X, tgt_X, pred_mask, tgt_mask)

    # Type loss: align each predicted slot to nearest target slot.
    if bool(pred_mask.any()) and bool(tgt_mask.any()):
        dists = torch.cdist(pred_X[pred_mask], tgt_X[tgt_mask], p=2)
        nn_idx = torch.min(dists, dim=1).indices
        tgt_types = target["type_target"][tgt_mask][nn_idx]
        logits_sel = pred_logits[pred_mask]
        type_loss = F.cross_entropy(logits_sel, tgt_types)
    else:
        type_loss = pred_X.new_tensor(0.0)

    # Extra point penalty for masked-but-empty targets.
    extra_penalty = pred_X.new_tensor(0.0)
    if bool(pred_mask.any()):
        with torch.no_grad():
            dists_extra = torch.cdist(pred_X[pred_mask], tgt_X[tgt_mask], p=2) if bool(tgt_mask.any()) else None
        if dists_extra is None or dists_extra.numel() == 0:
            extra_penalty = (pred_X[pred_mask] ** 2).mean()
        else:
            far_mask = (torch.min(dists_extra, dim=1).values > 1.0)
            if bool(far_mask.any()):
                extra_penalty = (pred_X[pred_mask][far_mask] ** 2).mean()

    return lambda_chamfer * cham + lambda_type * type_loss + lambda_extra * extra_penalty


def _build_learned_candidates(
    spec: CanonicalSpec,
    encoder: Optional[object],
    generator: Optional[BasisGenerator],
    n_learned: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ImageBasisElement]:
    if generator is None or n_learned <= 0:
        return []

    enc = encoder
    if enc is None:
        choice = os.getenv("EDE_IMAGES_GEO_ENCODER", "egnn").strip().lower()
        if choice in {"simple", "mlp"}:
            enc = SimpleGeoEncoder()
        else:
            try:
                enc = GeoEncoder()
            except Exception:
                enc = SimpleGeoEncoder()
    if enc is None:
        return []
    enc = enc.to(device=device)
    generator = generator.to(device=device)
    generator.eval()
    enc.eval()

    with torch.no_grad():
        z_global, charge_nodes, cond_nodes = enc.encode(
            spec, device=device, dtype=dtype
        )
        learned = generator(
            z_global=z_global,
            charge_nodes=charge_nodes,
            conductor_nodes=cond_nodes,
            n_candidates=n_learned,
        )

    # Attach stable group metadata for downstream sparsity.
    motif = 1
    safe: List[ImageBasisElement] = []
    for elem in learned:
        pos = elem.params.get("position")
        if pos is None or not torch.isfinite(pos).all():
            continue
        conductor_hint = 0
        try:
            conductor_hint = int(
                torch.as_tensor(elem.params.get("conductor_id", 0)).item()
            )
        except Exception:
            conductor_hint = 0
        annotate_group_info(
            elem,
            conductor_id=conductor_hint,
            family_name=elem.type,
            motif_index=motif,
        )
        motif += 1
        safe.append(elem)
    return safe[:n_learned]


def _select_basis_types(label: str, cfg: BilevelTrainConfig) -> List[str]:
    if label.startswith("sphere_dimer"):
        return cfg.basis_dimer
    if label.startswith("plane"):
        return cfg.basis_plane
    if label.startswith("sphere"):
        return cfg.basis_sphere
    return cfg.basis_sphere


def _build_candidates(
    spec: CanonicalSpec,
    label: str,
    cfg: BilevelTrainConfig,
    *,
    encoder: Optional[object],
    generator: Optional[BasisGenerator],
    device: torch.device,
    dtype: torch.dtype,
) -> List[ImageBasisElement]:
    basis_types = _select_basis_types(label, cfg)
    static = generate_candidate_basis(
        spec,
        basis_types=basis_types,
        n_candidates=cfg.n_candidates_static,
        device=device,
        dtype=dtype,
    )
    learned = _build_learned_candidates(
        spec,
        encoder=encoder,
        generator=generator,
        n_learned=cfg.n_candidates_learned,
        device=device,
        dtype=dtype,
    )
    return static + learned


def _make_operator(
    elems: Sequence[ImageBasisElement],
    points: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> BasisOperator:
    return BasisOperator(
        list(elems),
        points=points,
        device=device,
        dtype=dtype,
        row_weights=None,
    )


def _sample_stage0_task(
    cfg: BilevelTrainConfig,
    rng: np.random.Generator,
    *,
    base_plane: CanonicalSpec,
    base_sphere_ext: CanonicalSpec,
    base_sphere_int: CanonicalSpec,
) -> Tuple[str, CanonicalSpec, float]:
    q = float(rng.uniform(*cfg.ranges.q)) * float(rng.choice([-1.0, 1.0]))
    geom = rng.choice(cfg.stage0_geoms)
    if geom == "plane":
        z = float(rng.uniform(*cfg.ranges.plane_z))
        return "plane", _clone_spec_with_z(base_plane, z, q=q), z

    if geom == "sphere_internal":
        radius = float(base_sphere_int.conductors[0].get("radius", 1.0))
        frac = float(rng.uniform(*cfg.ranges.sphere_internal))
        z = frac * radius
        return "sphere_internal", _clone_spec_with_z(base_sphere_int, z, q=q), z

    # Default: external sphere
    radius = float(base_sphere_ext.conductors[0].get("radius", 1.0))
    frac = float(rng.uniform(*cfg.ranges.sphere_external))
    z = frac * radius
    return "sphere_external", _clone_spec_with_z(base_sphere_ext, z, q=q), z


def _sample_stage1_task(
    cfg: BilevelTrainConfig,
    rng: np.random.Generator,
    *,
    stage1_specs: Sequence[CanonicalSpec],
) -> Tuple[str, CanonicalSpec, float]:
    if not stage1_specs:
        raise RuntimeError("No Stage-1 specs provided.")
    spec = stage1_specs[int(rng.integers(0, len(stage1_specs)))]
    spheres = sorted(
        [c for c in getattr(spec, "conductors", []) if c.get("type") == "sphere"],
        key=lambda c: float(c.get("center", [0.0, 0.0, 0.0])[2]),
    )
    if len(spheres) < 2:
        raise RuntimeError("Stage-1 sampling expects a dimer of two spheres.")

    c0, c1 = spheres[0], spheres[1]
    z0 = float(c0.get("center", [0.0, 0.0, 0.0])[2])
    z1 = float(c1.get("center", [0.0, 0.0, 0.0])[2])
    r0 = float(c0.get("radius", 1.0))
    r1 = float(c1.get("radius", 1.0))

    gap = (z1 - r1) - (z0 + r0)
    margin = float(max(0.0, min(cfg.ranges_stage1.gap_margin, max(0.0, gap * 0.49))))
    z_min = z0 + r0 + margin
    z_max = z1 - r1 - margin
    if z_max <= z_min:
        z = 0.5 * (z_min + z_max)
    else:
        frac = float(rng.uniform(*cfg.ranges_stage1.charge_frac))
        frac = min(max(frac, 0.0), 1.0)
        z = z_min + frac * (z_max - z_min)

    q = float(rng.uniform(*cfg.ranges_stage1.q)) * float(rng.choice([-1.0, 1.0]))
    return "sphere_dimer", _clone_spec_with_z(spec, z, q=q), z


def _sample_task(
    cfg: BilevelTrainConfig,
    rng: np.random.Generator,
    *,
    base_plane: CanonicalSpec,
    base_sphere_ext: CanonicalSpec,
    base_sphere_int: CanonicalSpec,
    stage1_specs: Sequence[CanonicalSpec],
) -> Tuple[str, CanonicalSpec, float]:
    if cfg.stage >= 2:
        raise NotImplementedError("Stage-2 curriculum not yet implemented.")
    if cfg.stage >= 1:
        return _sample_stage1_task(cfg, rng, stage1_specs=stage1_specs)
    return _sample_stage0_task(
        cfg,
        rng,
        base_plane=base_plane,
        base_sphere_ext=base_sphere_ext,
        base_sphere_int=base_sphere_int,
    )


def _prepare_collocation(
    spec: CanonicalSpec,
    cfg: BilevelTrainConfig,
    *,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int,
    ratio_boundary: float,
    logger: JsonlLogger,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = get_collocation_data(
        spec=spec,
        logger=logger,
        device=device,
        dtype=dtype,
        return_is_boundary=True,
        rng=rng,
        n_points_override=n_points,
        ratio_override=ratio_boundary,
    )
    X, V, is_b = data  # get_collocation_data returns 3 tensors in this mode
    if X.numel() == 0 or V.numel() == 0:
        raise RuntimeError("Empty collocation batch.")
    return X, V, is_b


def _log_step(
    logger: JsonlLogger,
    step: int,
    loss: float,
    err_int: float,
    err_bc: float,
    sparsity: float,
    label_counts: Dict[str, int],
) -> None:
    logger.info(
        "bilevel_step",
        step=int(step),
        loss=float(loss),
        err_int=float(err_int),
        err_bc=float(err_bc),
        sparsity=float(sparsity),
        **{f"tasks_{k}": int(v) for k, v in label_counts.items()},
    )


def _save_checkpoint(
    out_dir: Path,
    step: int,
    lista: LISTALayer,
    generator: Optional[BasisGenerator],
    cfg: BilevelTrainConfig,
    *,
    geo_encoder: Optional[object] = None,
) -> None:
    payload = {
        "step": step,
        "config": asdict(cfg),
        "lista": lista.state_dict(),
    }
    if generator is not None:
        try:
            payload["basis_generator"] = generator.state_dict()
        except Exception:
            pass
    if geo_encoder is not None:
        try:
            payload["geo_encoder"] = geo_encoder.state_dict()  # type: ignore[arg-type]
        except Exception:
            try:
                payload["geo_encoder"] = geo_encoder  # type: ignore[assignment]
            except Exception:
                pass
    try:
        payload["lista_meta"] = {"K": int(getattr(lista, "K", 0))}
    except Exception:
        pass
    path = out_dir / f"checkpoint_step{step}.pt"
    torch.save(payload, path)


def train_diffusion_generator(cfg: DiffusionTrainingConfig) -> Dict[str, float]:
    """
    Skeleton training loop for the diffusion-based BasisGenerator.

    This is intentionally lightweight and CPU-friendly. It demonstrates how
    to assemble samples and run a single optimisation step; full training is
    deferred to future passes.
    """
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(str(out_dir / "train_diffusion_log.jsonl"))

    generator = DiffusionBasisGenerator(
        DiffusionGeneratorConfig(k_max=cfg.k_max, type_names=cfg.type_names)
    ).to(device=device)
    opt = torch.optim.Adam(generator.parameters(), lr=cfg.lr)

    # Placeholder data source: reuse a simple sphere spec with static candidates.
    spec = load_stage0_sphere_external()
    encoder = SimpleGeoEncoder()
    z_global, charge_nodes, cond_nodes = encoder.encode(spec, device=device, dtype=dtype)
    static_cands = _build_candidates(
        spec,
        label="sphere",
        cfg=BilevelTrainConfig(out_dir=cfg.out_dir),
        encoder=encoder,
        generator=None,
        device=device,
        dtype=dtype,
    )
    system = ImageSystem(static_cands, torch.ones(len(static_cands), device=device, dtype=dtype))
    sample = build_diffusion_training_sample_from_system(
        spec,
        system,
        z_global,
        torch.stack([c.embedding for c in charge_nodes], dim=0).mean(dim=0) if charge_nodes else torch.zeros_like(z_global),
        k_max=cfg.k_max,
        type_names=cfg.type_names,
        device=device,
        dtype=dtype,
    )

    metrics = {}
    for step in range(cfg.n_steps):
        opt.zero_grad(set_to_none=True)
        pred_elems = generator(
            z_global=z_global,
            charge_nodes=charge_nodes,
            conductor_nodes=cond_nodes,
            n_candidates=cfg.k_max,
        )
        if not pred_elems:
            logger.warning("Diffusion generator produced no elements; skipping step.")
            continue
        pos_pred = torch.stack([e.params["position"] for e in pred_elems], dim=0)
        logits_pred = torch.zeros(len(pred_elems), len(cfg.type_names), device=device, dtype=dtype)
        mask_pred = torch.ones(len(pred_elems), device=device, dtype=torch.bool)

        loss = diffusion_set_loss(
            pos_pred,
            logits_pred,
            mask_pred,
            sample,
            lambda_chamfer=cfg.lambda_chamfer,
            lambda_type=cfg.lambda_type,
            lambda_extra=cfg.lambda_extra,
        )
        loss.backward()
        opt.step()

        metrics = {"loss": float(loss.detach().item())}
        logger.info("Diffusion training step.", step=int(step), loss=float(loss.detach().item()))

    return metrics


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def train_stage0(cfg: BilevelTrainConfig) -> Dict[str, float]:
    """
    Stage-0/1 bilevel training loop with unrolled LISTA and BasisGenerator.

    ``cfg.stage`` selects the curriculum:
      - 0: Stage-0 (plane reflection + grounded sphere external/internal).
      - 1: Stage-1 (sphere dimer lens, gap charge along the axis).
      - 2: reserved for Stage-2 gratings (not implemented).

    The training loss is evaluated on a fresh validation collocation set per task,
    using the raw LISTA weights (no top-k or LS refit) to keep the outer objective
    differentiable.

    Invariants (Stage-0/1 head):
    - No augmented Lagrangian or adaptive collocation: collocation is single-pass
      via get_collocation_data, and aug_lagrange_cfg is intentionally absent.
    - LISTA-only optimisation: LISTALayer parameters are trained; encoder and
      BasisGenerator are run under no_grad()/.eval() and are not updated.
    - Evaluation-time safeguards (LISTA+ISTA fallback, top-k + LS refit) are
      deliberately excluded from the training head.
    """
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    if cfg.stage >= 2:
        raise NotImplementedError("Stage-2 curriculum not implemented; choose stage 0 or 1.")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(str(out_dir / "train_log.jsonl"))

    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    base_plane = _load_spec("plane_point.json")
    base_sphere_ext = load_stage0_sphere_external()
    base_sphere_int = _load_spec("sphere_axis_point_internal.json")
    base_stage1_specs: List[CanonicalSpec] = (
        _load_stage1_specs(cfg.stage1_include_variants) if cfg.stage >= 1 else []
    )

    generator: Optional[BasisGenerator] = (
        MLPBasisGenerator() if cfg.n_candidates_learned > 0 else None
    )
    encoder: Optional[object] = None
    if generator is not None:
        choice = os.getenv("EDE_IMAGES_GEO_ENCODER", "egnn").strip().lower()
        if choice in {"simple", "mlp"}:
            encoder = SimpleGeoEncoder()
        else:
            try:
                encoder = GeoEncoder()
            except Exception:
                encoder = SimpleGeoEncoder()

    # Warm-start candidate set to size the LISTA layer.
    label0, spec0, _ = _sample_task(
        cfg,
        np_rng,
        base_plane=base_plane,
        base_sphere_ext=base_sphere_ext,
        base_sphere_int=base_sphere_int,
        stage1_specs=base_stage1_specs,
    )
    cands0 = _build_candidates(
        spec0,
        label0,
        cfg,
        encoder=encoder,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if not cands0:
        raise RuntimeError("No candidates produced for initial task.")

    lista = LISTALayer(
        K=len(cands0),
        n_steps=cfg.lista_steps,
        rank=cfg.lista_rank,
        dense_threshold=cfg.lista_dense_threshold,
    ).to(device=device, dtype=dtype)

    params: List[Dict[str, object]] = [
        {"params": lista.parameters(), "lr": cfg.lr_lista},
    ]
    if generator is not None:
        params.append({"params": generator.parameters(), "lr": cfg.lr_geo})

    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    metrics_out: Dict[str, float] = {}
    for step in range(cfg.max_steps):
        opt.zero_grad(set_to_none=True)
        task_losses: List[torch.Tensor] = []
        err_int_terms: List[torch.Tensor] = []
        err_bc_terms: List[torch.Tensor] = []
        sparsities: List[torch.Tensor] = []
        label_counts: Dict[str, int] = {}

        for _ in range(cfg.batch_size):
            label, spec, z0 = _sample_task(
                cfg,
                np_rng,
                base_plane=base_plane,
                base_sphere_ext=base_sphere_ext,
                base_sphere_int=base_sphere_int,
                stage1_specs=base_stage1_specs,
            )
            label_counts[label] = label_counts.get(label, 0) + 1

            try:
                X_train, V_train, is_b_train = _prepare_collocation(
                    spec,
                    cfg,
                    rng=np_rng,
                    device=device,
                    dtype=dtype,
                    n_points=cfg.n_points_train,
                    ratio_boundary=cfg.ratio_boundary_train,
                    logger=logger,
                )
                X_val, V_val, is_b_val = _prepare_collocation(
                    spec,
                    cfg,
                    rng=np_rng,
                    device=device,
                    dtype=dtype,
                    n_points=cfg.n_points_val,
                    ratio_boundary=cfg.ratio_boundary_val,
                    logger=logger,
                )
            except Exception as exc:
                logger.warning(
                    "Collocation sampling failed; skipping task.",
                    error=str(exc),
                    label=label,
                    z0=float(z0),
                )
                continue

            candidates = _build_candidates(
                spec,
                label,
                cfg,
                encoder=encoder,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            if len(candidates) != len(cands0):
                logger.warning(
                    "Candidate count mismatch; skipping task.",
                    expected=int(len(cands0)),
                    actual=int(len(candidates)),
                    label=label,
                )
                continue

            op = _make_operator(candidates, X_train, device=device, dtype=dtype)
            g = V_train
            w = lista(
                op,
                X_train,
                g,
                group_ids=op.groups,
                lambda_group=cfg.lambda_group,
            )
            system = ImageSystem(candidates, w)
            V_pred = system.potential(X_val)

            mask_int = ~is_b_val
            err_int = _safe_mean((V_pred[mask_int] - V_val[mask_int]) ** 2)
            err_bc = _safe_mean(V_pred[is_b_val] ** 2)
            loss = err_int + cfg.lambda_bc * err_bc + cfg.lambda_l1 * w.abs().sum()

            task_losses.append(loss)
            err_int_terms.append(err_int.detach())
            err_bc_terms.append(err_bc.detach())
            sparsities.append((w.abs() > 1e-3).float().mean().detach())

        if not task_losses:
            logger.warning("No valid tasks in batch; continuing.")
            continue

        batch_loss = torch.stack(task_losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(lista.parameters(), cfg.grad_clip)
        if generator is not None:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
        opt.step()

        loss_val = float(batch_loss.detach().item())
        err_int_val = float(torch.stack(err_int_terms).mean().item())
        err_bc_val = float(torch.stack(err_bc_terms).mean().item())
        sparsity_val = float(torch.stack(sparsities).mean().item())
        _log_step(logger, step, loss_val, err_int_val, err_bc_val, sparsity_val, label_counts)

        metrics_out = {
            "loss": loss_val,
            "err_int": err_int_val,
            "err_bc": err_bc_val,
            "sparsity": sparsity_val,
        }

        if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
            _save_checkpoint(out_dir, step + 1, lista, generator, cfg, geo_encoder=encoder)

    # Save final checkpoint for downstream fine-tuning.
    _save_checkpoint(out_dir, cfg.max_steps, lista, generator, cfg, geo_encoder=encoder)
    return metrics_out


__all__ = [
    "BilevelTrainConfig",
    "Stage0Ranges",
    "train_stage0",
    "DiffusionTrainingConfig",
    "train_diffusion_generator",
]
