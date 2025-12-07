import torch

import electrodrive.learn.train as T


class FakeOOMModel(torch.nn.Module):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.lin = torch.nn.Linear(3, 1)
        self.last_batch = 0

    def compute_loss(self, batch, loss_weights):
        m = batch["X"].shape[0]
        self.last_batch = m
        if m > self.threshold:
            raise RuntimeError("Simulated OOM")
        out = self.lin(batch["X"])
        loss = (out ** 2).mean()
        return {"total": loss}


def _patch_autotune_cuda(monkeypatch, model, mem_state, mem_per_point):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(T, "_select_autocast_dtype", lambda device: (False, None, "off"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "FakeGPU", raising=False)

    def fake_mem_get_info(device=None):
        return (mem_state["free"], mem_state["total"])

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info, raising=False)

    class _Props:
        total_memory = mem_state["total"]

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props(), raising=False)

    def fake_max_memory_allocated(device=None):
        return int(getattr(model, "last_batch", 0) * mem_per_point)

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake_max_memory_allocated, raising=False)

    real_zeros = torch.zeros
    real_ones = torch.ones

    def cpu_zeros(*shape, device=None, **kwargs):
        return real_zeros(*shape, device=None, **kwargs)

    def cpu_ones(*shape, device=None, **kwargs):
        return real_ones(*shape, device=None, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "ones", cpu_ones)


def test_autotune_skips_oom_candidates(monkeypatch):
    model = FakeOOMModel(threshold=4096)
    mem_state = {"free": 10 * 1024**3, "total": 16 * 1024**3}
    mem_per_point = 1_000_000  # bytes per point in the fake peak reading

    T._AUTOTUNE_POINTS_CACHE.clear()
    _patch_autotune_cuda(monkeypatch, model, mem_state, mem_per_point)

    pts = T._autotune_points_per_step(model, torch.device("cuda:0"))

    assert pts > 0
    assert pts <= model.threshold


def test_autotune_shrinks_with_low_vram(monkeypatch):
    model = FakeOOMModel(threshold=1000000)
    mem_state = {"free": 8 * 1024**3, "total": 16 * 1024**3}
    mem_per_point = 150_000  # bytes per point for peak memory simulation

    T._AUTOTUNE_POINTS_CACHE.clear()
    _patch_autotune_cuda(monkeypatch, model, mem_state, mem_per_point)
    device = torch.device("cuda:0")

    high_free_pts = T._autotune_points_per_step(model, device)

    T._AUTOTUNE_POINTS_CACHE.clear()
    mem_state["free"] = 2 * 1024**3
    low_free_pts = T._autotune_points_per_step(model, device)

    assert low_free_pts > 0
    assert high_free_pts >= low_free_pts
    assert low_free_pts < high_free_pts
