import torch

from electrodrive.gfdsl.compile import CompileContext


def test_compile_context_fields_and_clone():
    ctx = CompileContext()
    assert isinstance(ctx.device, torch.device)
    assert isinstance(ctx.dtype, torch.dtype)
    assert ctx.eval_backend in {"dense", "operator", "hybrid"}
    assert isinstance(ctx.cache, dict)

    # extras must exist and be mutable
    assert isinstance(ctx.extras, dict)
    ctx.extras["foo"] = "bar"
    assert ctx.extras["foo"] == "bar"

    clone = ctx.clone_with(eval_backend="operator")
    assert isinstance(clone, CompileContext)
    assert clone.eval_backend == "operator"
    assert clone.device == ctx.device
    assert clone.dtype == ctx.dtype
    assert clone.cache is ctx.cache
    assert clone.extras is not ctx.extras
    assert clone.extras.get("foo") == "bar"
