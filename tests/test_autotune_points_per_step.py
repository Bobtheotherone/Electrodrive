import torch

import electrodrive.learn.train as T


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(3, 1)

    def compute_loss(self, batch, loss_weights):
        out = self.l(batch["X"])
        loss = (out ** 2).mean()
        return {"total": loss}


def test_autotune_basic_bounds_cpu():
    T._AUTOTUNE_POINTS_CACHE.clear()
    model = DummyModel()
    pts = T._autotune_points_per_step(model, torch.device("cpu"))
    max_candidate = max(getattr(T, "_AUTOTUNE_CANDIDATES", [32768]))
    assert pts > 0
    assert pts <= max_candidate


def test_autotune_monotonic_with_free_mem(monkeypatch):
    T._AUTOTUNE_POINTS_CACHE.clear()
    model = DummyModel()

    # Pretend CUDA exists and keep allocations on CPU for safety.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(T, "_select_autocast_dtype", lambda device: (False, None, "off"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 5 * 1024**2, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "FakeGPU", raising=False)

    total_bytes = 24 * 1024**3

    class _Props:
        total_memory = total_bytes

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props(), raising=False)

    free_state = {"free": 4 * 1024**3}

    def fake_mem_get_info(device=None):
        return (free_state["free"], total_bytes)

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info, raising=False)

    real_zeros = torch.zeros
    real_ones = torch.ones

    def cpu_zeros(*shape, device=None, **kwargs):
        return real_zeros(*shape, device=None, **kwargs)

    def cpu_ones(*shape, device=None, **kwargs):
        return real_ones(*shape, device=None, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "ones", cpu_ones)

    device = torch.device("cuda:0")
    pts_low = T._autotune_points_per_step(model, device)

    # Increase available VRAM and expect non-decreasing autotuned size.
    T._AUTOTUNE_POINTS_CACHE.clear()
    free_state["free"] = 16 * 1024**3
    pts_high = T._autotune_points_per_step(model, device)

    max_candidate = max(getattr(T, "_AUTOTUNE_CANDIDATES", [pts_high]))
    assert pts_low > 0
    assert pts_high >= pts_low
    assert pts_high <= max_candidate
