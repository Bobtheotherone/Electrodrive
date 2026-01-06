import torch

from electrodrive.verify.green_decomposition import GreenDecomposition


def test_green_decomposition_defaults_to_singular() -> None:
    def singular(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x - y, dim=1)

    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    decomp = GreenDecomposition(singular)
    out = decomp(x, y)

    assert torch.allclose(out, singular(x, y))


def test_green_decomposition_adds_smooth() -> None:
    def singular(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * 2.0

    def smooth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * 0.5

    x = torch.zeros(3, 3)
    y = torch.zeros(3, 3)

    decomp = GreenDecomposition(singular, smooth)
    out = decomp(x, y)

    assert torch.allclose(out, torch.full((3,), 2.5))
