import torch

from electrodrive.learn.specs import ExperimentConfig, ModelSpec
from electrodrive.learn.encoding import ENCODING_DIM
from electrodrive.learn.train import initialize_model


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_pinn_large_has_more_params_than_default():
    # Default pinn_harmonic config (backwards-compatible)
    base_cfg = ExperimentConfig(
        exp_name="test_base",
    )
    base_cfg.model = ModelSpec(
        model_type="pinn_harmonic",
        params={},
    )

    # Large variant config
    large_cfg = ExperimentConfig(
        exp_name="test_large",
    )
    large_cfg.model = ModelSpec(
        model_type="pinn_harmonic_large",
        params={},
    )

    base_model = initialize_model(base_cfg)
    large_model = initialize_model(large_cfg)

    base_n = _count_params(base_model)
    large_n = _count_params(large_model)

    assert large_n > base_n


def test_pinn_large_forward_smoke():
    # Minimal config for large model
    cfg = ExperimentConfig(
        exp_name="test_large_forward",
    )
    cfg.model = ModelSpec(
        model_type="pinn_harmonic_large",
        params={},
    )

    model = initialize_model(cfg)
    model.eval()  # no checkpointing in eval

    # Tiny CPU-only batch: 8 points, 3D coords, plus encoding
    device = torch.device("cpu")
    x = torch.zeros(8, 3, device=device, dtype=torch.float32)

    # Dummy encoding vector with correct width; model will broadcast it
    # across the batch when encoding.dim() == 1.
    encoding = torch.zeros(
        ENCODING_DIM,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        out = model(x, encoding)

    # Shape: [N, 1]
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == 1
    assert torch.isfinite(out).all()