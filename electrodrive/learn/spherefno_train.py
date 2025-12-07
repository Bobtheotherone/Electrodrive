from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from electrodrive.learn.datasets_stage0 import (
    Stage0SphereAxisDataset,
    Stage0SphereRanges,
)
from electrodrive.learn.neural_operators import SphereFNO


def _default_device(config_device: str) -> torch.device:
    if config_device:
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _resolve_ranges(cfg: Dict[str, Any]) -> Stage0SphereRanges:
    data = cfg.get("dataset", {}).get("ranges", {})
    return Stage0SphereRanges(
        q=tuple(data.get("q", (0.5, 2.0))),
        a=tuple(data.get("a", (0.5, 2.0))),
        z0_over_a=tuple(data.get("z0_over_a", (1.05, 3.0))),
        center=tuple(data.get("center", (0.0, 0.0, 0.0))),
    )


def _make_datasets(cfg: Dict[str, Any], dtype: torch.dtype) -> Tuple[Stage0SphereAxisDataset, Stage0SphereAxisDataset]:
    ds_cfg = cfg.get("dataset", {})
    n_train = int(ds_cfg.get("n_train", 10000))
    n_val = int(ds_cfg.get("n_val", 1000))
    n_theta = int(ds_cfg.get("n_theta", 64))
    n_phi = int(ds_cfg.get("n_phi", 128))
    seed = int(ds_cfg.get("seed", 0))
    ranges = _resolve_ranges(cfg)

    full_ds = Stage0SphereAxisDataset(
        n_samples=n_train + n_val,
        ranges=ranges,
        n_theta=n_theta,
        n_phi=n_phi,
        seed=seed,
        dtype=dtype,
    )
    # Deterministic split
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_ds,
        lengths=[n_train, n_val],
        generator=gen,
    )
    return train_ds, val_ds


def _make_model(cfg: Dict[str, Any]) -> SphereFNO:
    mcfg = cfg.get("model", {})
    return SphereFNO(
        n_theta=int(mcfg.get("n_theta", 64)),
        n_phi=int(mcfg.get("n_phi", 128)),
        modes_theta=int(mcfg.get("modes_theta", 16)),
        modes_phi=int(mcfg.get("modes_phi", 16)),
        width=int(mcfg.get("width", 64)),
        n_layers=int(mcfg.get("n_layers", 4)),
        param_hidden=int(mcfg.get("param_hidden", 64)),
    )


def _compute_rel_errors(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    rel_l2 = float(torch.linalg.norm(diff) / torch.linalg.norm(target).clamp_min(1e-12))
    rel_linf = float(torch.max(torch.abs(diff)) / torch.max(torch.abs(target)).clamp_min(1e-12))
    return {"val_rel_l2": rel_l2, "val_rel_linf": rel_linf}


def train(config: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _default_device(config.get("device", ""))
    dtype = getattr(torch, config.get("dtype", "float32"))

    train_ds, val_ds = _make_datasets(config, dtype=dtype)
    batch_size = int(config.get("training", {}).get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = _make_model(config).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("training", {}).get("lr", 1e-3)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 0.0)),
    )

    max_epochs = int(config.get("training", {}).get("epochs", 10))
    grad_clip = float(config.get("training", {}).get("grad_clip_norm", 0.0))
    best_path = out_dir / "best.pt"
    best_l2 = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            params = batch["params"].to(device=device, dtype=dtype)
            target = batch["V"].to(device=device, dtype=dtype)
            pred = model(params)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            rel_errors = []
            for batch in val_loader:
                params = batch["params"].to(device=device, dtype=dtype)
                target = batch["V"].to(device=device, dtype=dtype)
                pred = model(params)
                val_losses.append(F.mse_loss(pred, target).item())
                rel_errors.append(_compute_rel_errors(pred, target))

        mean_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        # Average relative errors over batches
        if rel_errors:
            val_rel_l2 = float(sum(r["val_rel_l2"] for r in rel_errors) / len(rel_errors))
            val_rel_linf = float(sum(r["val_rel_linf"] for r in rel_errors) / len(rel_errors))
        else:
            val_rel_l2 = float("inf")
            val_rel_linf = float("inf")

        if val_rel_l2 < best_l2:
            best_l2 = val_rel_l2
            payload = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_rel_l2": val_rel_l2,
                "val_rel_linf": val_rel_linf,
            }
            torch.save(payload, best_path)

        print(
            f"[epoch {epoch}/{max_epochs}] train_loss={total_loss/ max(1,len(train_loader)):.4e} "
            f"val_loss={mean_val_loss:.4e} val_rel_l2={val_rel_l2:.3e} val_rel_linf={val_rel_linf:.3e}"
        )

    print(f"Best checkpoint: {best_path} (val_rel_l2={best_l2:.3e})")
    return best_path


def main(config_path: Path, out_dir: Path) -> Path:
    cfg = _load_config(config_path)
    best_path = train(cfg, out_dir)
    print(json.dumps({"best_checkpoint": str(best_path)}, indent=2))
    return best_path
