import torch
import torch.nn as nn


class LaplaceLoss(nn.Module):
    def forward(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # u: [N,1], x: [N,3] (requires_grad=True)
        grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        ux, uy, uz = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        uyy = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(uy),
                                  create_graph=True, retain_graph=True)[0][:, 1:2]
        uzz = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(uz),
                                  create_graph=True, retain_graph=True)[0][:, 2:2+1]
        lap = uxx + uyy + uzz
        return torch.mean(lap ** 2)


class BCLoss(nn.Module):
    def forward(self, V_pred: torch.Tensor, V_target: torch.Tensor) -> torch.Tensor:
        return torch.mean((V_pred - V_target) ** 2)


def compute_dynamic_weights(epoch: int, total_epochs: int, w_pde_init=1.0, w_bc_init=100.0) -> tuple[float, float]:
    progress = float(epoch) / max(1, total_epochs)
    w_bc = w_bc_init * (0.1 ** progress)  # decay BC weight over time
    w_pde = w_pde_init
    return w_pde, w_bc




