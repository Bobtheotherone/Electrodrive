import torch
import electrodrive.learn.train as T

def test_select_autocast_dtype_cpu_off():
    use, dt, mode = T._select_autocast_dtype(torch.device("cpu"))
    assert use is False and dt is None and mode == "off"

def test_select_autocast_dtype_cuda_bf16(monkeypatch):
    # Patch methods on torch.cuda instead of replacing the module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    use, dt, mode = T._select_autocast_dtype(torch.device("cuda:0"))
    assert use is True and mode == "bf16" and (dt is torch.bfloat16)

def test_maybe_compile_noop_without_feature(monkeypatch):
    model = torch.nn.Linear(3, 1)
    # Simulate environments without torch.compile
    if hasattr(torch, "compile"):
        monkeypatch.delattr(torch, "compile", raising=False)
    out = T._maybe_compile(model, compile_flag=True, mode="reduce-overhead")
    assert out is model  # soft no-op
