import types
import torch
import numpy as np
import electrodrive.learn.train as T

import copy
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
    def compute_loss(self, batch, loss_weights):
        x = batch["X"]
        total = (x.mean() + self.w).pow(2)
        return {"total": total, "pde": total, "bc": total}

def _fake_build_dataloaders(config):
    # Train: two batches so we test slicing + accumulation + leftover flush
    def make(n):
        return {"X": torch.randn(n, 3), "V_gt": torch.zeros(n)}
    train_loader = [make(1000), make(700)]
    # Val: one small batch to exercise validate()
    val_loader = [make(32)]
    return train_loader, val_loader

def test_train_microbatch_accumulation_and_validation(tmp_path, monkeypatch):
    # Quiet logger
    class _L:
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
        def warning(self, *a, **k): pass
    monkeypatch.setattr(T, "logger", _L())

    monkeypatch.setattr(T, "initialize_model", lambda cfg: DummyModel())
    monkeypatch.setattr(T, "build_dataloaders", _fake_build_dataloaders)

    cfg = types.SimpleNamespace(
        device="cpu",
        seed=123,
        train_dtype="float32",
        model=types.SimpleNamespace(model_type="pinn_harmonic", params={}),
        trainer=types.SimpleNamespace(
            learning_rate=1e-3, weight_decay=0.0, max_epochs=1,
            lr_scheduler="none", loss_weights={"total":1.0},
            log_every_n_steps=1, val_every_n_epochs=1,
            ckpt_every_n_epochs=1000, early_stopping_patience=0,
            grad_clip_norm=0.0, amp=None, compile=False,
            accum_steps=2, points_per_step=256
        ),
        dataset={},
        curriculum={},
        evaluation={}
    )
    rc = T.train(cfg, tmp_path)
    assert rc == 0

    mfile = tmp_path / "metrics.jsonl"
    assert mfile.exists()
    lines = [ln.strip() for ln in mfile.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any('"val_total"' in ln for ln in lines)


def test_microbatch_boundary_coverage():
    torch.manual_seed(123)
    np.random.seed(123)

    N = 128
    boundary_start = int(0.8 * N)
    is_boundary = torch.zeros(N, dtype=torch.bool)
    is_boundary[boundary_start:] = True

    batch = {
        "X": torch.randn(N, 3),
        "V_gt": torch.randn(N),
        "is_boundary": is_boundary,
        "mask_finite": torch.ones(N, dtype=torch.bool),
    }

    points_per_step = 32
    boundary_fraction = 0.1

    sub = T._subsample_collocation_batch(
        batch, points_per_step=points_per_step, boundary_fraction=boundary_fraction
    )

    assert sub["X"].shape == (points_per_step, 3)
    assert sub["V_gt"].shape == (points_per_step,)
    assert sub["is_boundary"].shape == (points_per_step,)
    assert sub["mask_finite"].shape == (points_per_step,)
    assert bool(sub["is_boundary"].any())


def test_microbatch_without_boundary_key():
    torch.manual_seed(7)

    N = 50
    batch = {
        "X": torch.randn(N, 3),
        "V_gt": torch.randn(N),
    }
    points_per_step = 10

    sub = T._subsample_collocation_batch(
        batch, points_per_step=points_per_step, boundary_fraction=0.5
    )

    assert sub["X"].shape[0] == points_per_step
    assert sub["V_gt"].shape[0] == points_per_step


def test_microbatch_noop_and_clamped_fraction():
    torch.manual_seed(99)
    np.random.seed(99)

    N = 16
    is_boundary = torch.zeros(N, dtype=torch.bool)
    # Make sure we have more boundary points than the microbatch needs.
    is_boundary[4:] = True

    batch = {
        "X": torch.randn(N, 3),
        "V_gt": torch.randn(N),
        "is_boundary": is_boundary,
    }

    # points_per_step >= N should be a no-op regardless of boundary_fraction magnitude
    same = T._subsample_collocation_batch(batch, points_per_step=N, boundary_fraction=5.0)
    assert same is batch

    # Extreme boundary_fraction should be clamped to 1.0, saturating with boundary points
    points_per_step = 8
    sub = T._subsample_collocation_batch(
        batch, points_per_step=points_per_step, boundary_fraction=1.5
    )
    assert sub["X"].shape[0] == points_per_step
    assert sub["is_boundary"].sum().item() == points_per_step


def test_accumulation_scaling_equivalence():
    torch.manual_seed(0)

    class TinyNet(nn.Module):
        def __init__(self, in_features=4, hidden=8, out_features=2):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)
            self.fc2 = nn.Linear(hidden, out_features)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    in_features = 4
    hidden = 8
    out_features = 2

    base_model = TinyNet(in_features=in_features, hidden=hidden, out_features=out_features)
    accum_model = copy.deepcopy(base_model)

    points_per_step = 16
    accum_steps = 4
    total = points_per_step * accum_steps

    x = torch.randn(total, in_features)
    y = torch.randint(0, out_features, (total,))

    # Scenario 1: single large batch
    opt_large = torch.optim.SGD(base_model.parameters(), lr=0.1)
    opt_large.zero_grad()
    logits_large = base_model(x)
    loss_large = F.cross_entropy(logits_large, y)
    loss_large.backward()
    grads_large = torch.cat(
        [p.grad.view(-1) for p in base_model.parameters() if p.grad is not None]
    )

    # Scenario 2: accumulated microbatches with scaled loss
    opt_accum = torch.optim.SGD(accum_model.parameters(), lr=0.1)
    opt_accum.zero_grad()
    for i in range(accum_steps):
        start = i * points_per_step
        end = (i + 1) * points_per_step
        xb = x[start:end]
        yb = y[start:end]
        logits_micro = accum_model(xb)
        loss_micro = F.cross_entropy(logits_micro, yb)
        loss_micro = loss_micro / float(accum_steps)
        loss_micro.backward()

    grads_accum = torch.cat(
        [p.grad.view(-1) for p in accum_model.parameters() if p.grad is not None]
    )

    assert torch.allclose(grads_large, grads_accum, atol=1e-6, rtol=1e-4)
    cos = F.cosine_similarity(grads_large, grads_accum, dim=0)
    assert float(cos) > 0.999
