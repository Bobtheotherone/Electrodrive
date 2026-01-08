import pytest
import torch

from electrodrive.images.learned_solver import _group_prox as learned_group_prox
from electrodrive.images.search import _group_prox as search_group_prox
from electrodrive.images.optim.grouplasso_implicit import _group_prox as implicit_group_prox


def _group_prox_reference(
    w: torch.Tensor,
    group_ids: torch.Tensor,
    lambda_group: float | torch.Tensor,
) -> torch.Tensor:
    if group_ids.numel() == 0:
        return w
    w_out = w.clone()
    group_ids = group_ids.to(device=w.device)
    if torch.is_tensor(lambda_group):
        lam_vec = lambda_group.to(device=w.device, dtype=w.dtype).view(-1)
        if lam_vec.numel() == 1:
            lam_vec = lam_vec.expand_as(w)
    else:
        lam_vec = None
    unique_groups = torch.unique(group_ids)
    for g_val in unique_groups:
        mask = group_ids == g_val
        if not bool(mask.any()):
            continue
        w_g = w_out[mask]
        norm_g = torch.linalg.norm(w_g)
        lam = float(lam_vec[mask].mean().item()) if lam_vec is not None else float(lambda_group)
        if float(norm_g) <= lam:
            w_out[mask] = 0.0
        else:
            shrink = (norm_g - lam) / norm_g
            w_out[mask] = shrink * w_g
    return w_out


def _run_group_prox_case(device: torch.device) -> None:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    n = 32
    w = torch.randn(n, device=device)
    group_ids = torch.randint(0, 6, (n,), device=device)

    lambda_scalar = 0.2
    ref_scalar = _group_prox_reference(w, group_ids, lambda_scalar)
    out_implicit = implicit_group_prox(w, group_ids, lambda_scalar)
    out_learned = learned_group_prox(w, group_ids, lambda_scalar)
    out_search = search_group_prox(w, group_ids, lambda_scalar)

    assert torch.allclose(out_implicit, ref_scalar)
    assert torch.allclose(out_learned, ref_scalar)
    assert torch.allclose(out_search, ref_scalar)

    lam_vec = torch.rand(n, device=device) * 0.3
    ref_vec = _group_prox_reference(w, group_ids, lam_vec)
    out_implicit_vec = implicit_group_prox(w, group_ids, lam_vec)
    out_learned_vec = learned_group_prox(w, group_ids, lam_vec)
    out_search_vec = search_group_prox(w, group_ids, lam_vec)

    assert torch.allclose(out_implicit_vec, ref_vec)
    assert torch.allclose(out_learned_vec, ref_vec)
    assert torch.allclose(out_search_vec, ref_vec)


def test_group_prox_vectorized_cpu() -> None:
    _run_group_prox_case(torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for group prox test"
)
def test_group_prox_vectorized_cuda() -> None:
    _run_group_prox_case(torch.device("cuda"))
