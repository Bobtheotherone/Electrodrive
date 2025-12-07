# electrodrive/core/pinn.py
"""
Physics-Informed Neural Network (PINN) solver for electrostatics.

This module is imported by tests (see test_pinn_train_eval_smoke.py) and
by higher-level CLI/orchestration code. It is designed to be:

- Import-safe: works even if some optional dependencies are missing,
  using soft fallbacks instead of failing at import time.
- Numerically conservative: model weights in FP32/FP64 as configured;
  compatible with external AMP/autocast (we do not force AMP here).
- Minimal but complete: provides pinn_train_eval, pinn_train_eval_stub,
  and pinn_training_step with stable semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from electrodrive.utils.logging import JsonlLogger
from electrodrive.utils.config import PINNConfig, K_E
from electrodrive.orchestration.parser import CanonicalSpec
from electrodrive.core.pinn_model import FourierMLP
from electrodrive.core.pinn_loss import LaplaceLoss, BCLoss, compute_dynamic_weights

# -----------------------------------------------------------------------------
# Optional pinn_data samplers with safe fallback
# -----------------------------------------------------------------------------
try:
    from electrodrive.core.pinn_data import InteriorSampler, BoundarySampler
except Exception as _pinn_err:  # pragma: no cover - defensive import guard

    class InteriorSampler:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise RuntimeError(
                "PINN backend is disabled (missing or broken electrodrive.core.pinn_data)."
            ) from _pinn_err

        def sample(self, *_, **__):
            raise RuntimeError("PINN backend is disabled.") from _pinn_err

    class BoundarySampler:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise RuntimeError(
                "PINN backend is disabled (missing or broken electrodrive.core.pinn_data)."
            ) from _pinn_err

        def sample(self, *_, **__):
            raise RuntimeError("PINN backend is disabled.") from _pinn_err


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------


def _init_device(cfg: PINNConfig) -> torch.device:
    """
    Choose compute device based on cfg and CUDA availability.

    Conservative: if CUDA is unavailable or cfg.use_gpu is False,
    falls back to CPU without raising.
    """
    if getattr(cfg, "use_gpu", True) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Main training + evaluation
# -----------------------------------------------------------------------------


def pinn_train_eval(
    spec: CanonicalSpec,
    cfg: PINNConfig,
    logger: JsonlLogger,
) -> Dict[str, Any]:
    """
    Train a simple PINN for the given canonical problem specification.

    Returns a dictionary containing, at minimum:
    - "solution": an object with:
        - eval(x,y,z) -> float potential
        - eval_V_E_batched(P) -> (V[N], E[N,3]) with E possibly NaNs
    - "boundary_samples": list of boundary potentials used for diagnostics
    - "bc_rmse": float RMSE of boundary condition on a held-out sample
    - plus some training diagnostics.

    Notes:
    - We keep the model in FP32 or FP64 (cfg.fp64) for stability.
    - External code may wrap calls in autocast if desired; we do not
      hardcode AMP inside this function.
    """
    # Dtype and device
    base_dtype = torch.float64 if getattr(cfg, "fp64", False) else torch.float32
    torch.set_default_dtype(base_dtype)
    device = _init_device(cfg)

    # Seeding
    seed = int(getattr(cfg, "seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1) Singular potential from explicit point charges
    # ------------------------------------------------------------------
    src_pos: list[torch.Tensor] = []
    src_q: list[torch.Tensor] = []
    for ch in getattr(spec, "charges", []):
        if ch.get("type") == "point":
            src_pos.append(
                torch.tensor(
                    ch["pos"],
                    dtype=torch.float64,
                    device=device,
                )
            )
            src_q.append(
                torch.tensor(
                    float(ch["q"]),
                    dtype=torch.float64,
                    device=device,
                )
            )

    def V_singular(x: torch.Tensor) -> torch.Tensor:
        """
        Potential from discrete charges at locations src_pos with magnitudes src_q.

        x: [N, 3] in any floating dtype (we cast internally as needed).
        """
        if not src_pos:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        dtype = x.dtype
        out = torch.zeros(x.shape[0], dtype=dtype, device=x.device)
        for p64, q64 in zip(src_pos, src_q):
            p = p64.to(device=x.device, dtype=dtype)
            q = q64.to(device=x.device, dtype=dtype)
            r = torch.linalg.norm(x - p[None, :], dim=1).clamp_min(1e-12)
            out += (K_E * q) / r
        return out

    # ------------------------------------------------------------------
    # 2) Network approximating u(x) such that V(x) = V_singular(x) + u(x)
    # ------------------------------------------------------------------
    width = int(getattr(cfg, "width", getattr(cfg, "hidden_dim", 64)))
    depth = int(getattr(cfg, "depth", getattr(cfg, "layers", 5)))

    model = FourierMLP(
        input_dim=3,
        hidden_dim=width,
        num_layers=depth,
        fourier_features=True,
        fourier_scale=10.0,
    ).to(device=device, dtype=base_dtype)

    logger.info(
        "PINN model initialized.",
        extra={
            "params": int(sum(p.numel() for p in model.parameters())),
            "fp64": bool(getattr(cfg, "fp64", False)),
            "width": width,
            "depth": depth,
        },
    )

    # ------------------------------------------------------------------
    # 3) Data samplers (Interior + Boundary)
    # ------------------------------------------------------------------
    # For basic examples: domain wide enough around origin.
    # If there is a conducting plane, start z slightly above 0 to avoid degeneracy.
    has_plane = any(
        c.get("type") == "plane"
        for c in getattr(spec, "conductors", [])
    )
    zmin = 0.01 if has_plane else -1.5
    domain = [[-1.5, 1.5], [-1.5, 1.5], [zmin, 1.5]]

    try:
        interior = InteriorSampler(domain=domain, seed=seed)
        boundary = BoundarySampler(plane_L=2.0, seed=seed + 1)
    except RuntimeError as e:
        msg = (
            "PINN initialization failed due to missing/broken pinn_data "
            f"samplers: {e}"
        )
        logger.error("PINN initialization error.", extra={"error": str(e)})
        return {
            "error": msg,
            "bc_rmse": float("nan"),
            "solution": None,
            "boundary_samples": [],
        }

    # ------------------------------------------------------------------
    # 4) Losses, optimizer, scheduler
    # ------------------------------------------------------------------
    pde_loss = LaplaceLoss()
    bc_loss = BCLoss()

    lr = float(getattr(cfg, "learning_rate", getattr(cfg, "lr", 1e-3)))
    opt = optim.Adam(model.parameters(), lr=lr)

    # NOTE: Some PyTorch versions error on unknown kwargs (e.g., verbose).
    # Keep the call maximally compatible.
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=100,
    )

    # Training hyperparameters with safe defaults
    n_collocation = int(getattr(cfg, "n_collocation", 2048))
    n_boundary = int(getattr(cfg, "n_boundary", 512))
    max_epochs = int(getattr(cfg, "epochs", 800))
    early_patience = int(getattr(cfg, "early_stop_patience", 200))
    w_pde_init = float(getattr(cfg, "w_pde", 1.0))
    w_bc_init = float(getattr(cfg, "w_bc", 1.0))

    best = float("inf")
    patience = 0
    history: Dict[str, list[float]] = {
        "train_pde": [],
        "train_bc": [],
        "total": [],
    }

    logger.info(
        "Starting PINN training...",
        extra={
            "epochs": max_epochs,
            "n_collocation": n_collocation,
            "n_boundary": n_boundary,
            "lr": lr,
        },
    )

    # ------------------------------------------------------------------
    # 5) Training loop (short, robust; not aiming for high accuracy)
    # ------------------------------------------------------------------
    for epoch in range(max_epochs):
        model.train()

        # Interior collocation points
        X_int = interior.sample(
            n=n_collocation,
            device=device,
            dtype=base_dtype,
        ).requires_grad_(True)

        # Boundary points; homogeneous Dirichlet on the sampled boundary
        X_bc = boundary.sample(
            n=n_boundary,
            device=device,
            dtype=base_dtype,
        )
        V_bc_target = torch.zeros(
            X_bc.shape[0],
            device=device,
            dtype=base_dtype,
        )

        # Network outputs and potentials
        u_int = model(X_int)
        u_bc = model(X_bc)
        V_bc_pred = V_singular(X_bc).unsqueeze(1) + u_bc

        # Losses
        loss_p = pde_loss(u_int, X_int)
        loss_b = bc_loss(V_bc_pred.squeeze(), V_bc_target)

        w_pde, w_bc = compute_dynamic_weights(
            epoch,
            max_epochs,
            w_pde_init,
            w_bc_init,
        )
        loss = w_pde * loss_p + w_bc * loss_b

        # Optimizer step
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        try:
            sched.step(loss)
        except Exception:
            # Scheduler is best-effort; ignore numerical issues.
            pass

        # Lightweight logging every 100 epochs
        if epoch % 100 == 0:
            lp = float(loss_p.item())
            lb = float(loss_b.item())
            lt = float(loss.item())
            history["train_pde"].append(lp)
            history["train_bc"].append(lb)
            history["total"].append(lt)
            logger.info(
                "PINN epoch",
                extra={
                    "epoch": epoch,
                    "loss_total": lt,
                    "loss_pde": lp,
                    "loss_bc": lb,
                },
            )

        # Early stopping on plateau of total loss
        cur = float(loss.item())
        if cur < best - 1e-6:
            best = cur
            patience = 0
        else:
            patience += 1

        if early_patience > 0 and patience >= early_patience:
            logger.info(
                "PINN early stopping.",
                extra={
                    "epoch": epoch,
                    "best_loss": float(best),
                },
            )
            break

    # ------------------------------------------------------------------
    # 6) Final boundary evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        Xb = boundary.sample(
            n=n_boundary,
            device=device,
            dtype=base_dtype,
        )
        u_b = model(Xb)
        Vb_pred = V_singular(Xb).unsqueeze(1) + u_b
        Vb_flat = Vb_pred.squeeze()
        Vb_target = torch.zeros_like(Vb_flat)

        bc_rmse = (
            torch.mean((Vb_flat - Vb_target) ** 2)
            .sqrt()
            .item()
        )

        # Store as plain floats for JSON compatibility
        boundary_samples = (
            Vb_flat.detach()
            .cpu()
            .to(torch.float64)
            .numpy()
            .tolist()
        )

    logger.info("PINN training done.", extra={"bc_rmse": float(bc_rmse)})

    # ------------------------------------------------------------------
    # 7) Solution wrapper with safeguards
    # ------------------------------------------------------------------
    class PINNSolution:
        """
        Lightweight solution wrapper.

        Methods:
        - eval((x,y,z)) -> float potential
        - eval_V_E_batched(P) -> (V[N], E[N,3])
          where E is NaN (PINN E-field not implemented here).

        Numeric safeguards:
        - Inputs are clamped to a reasonable domain (matching training box)
          to avoid absurd extrapolation.
        - Outputs are clamped to a safe magnitude to avoid inf/NaN leaking
          into energy consistency checks.
        """

        def __init__(
            self,
            model_: nn.Module,
            V_singular_fn,
            device_: torch.device,
            dtype_: torch.dtype,
        ) -> None:
            self.model = model_
            self.V_singular = V_singular_fn
            self._device = device_
            self._dtype = dtype_
            self.meta: Dict[str, Any] = {
                "mode": "pinn",
            }
            # Domain and magnitude clamps (chosen conservatively)
            self._xyz_limit = 5.0
            self._v_abs_max = float(K_E) * 10.0  # generous but finite

        def _clamp_points(self, P: torch.Tensor) -> torch.Tensor:
            limit = self._xyz_limit
            return torch.clamp(P, -limit, limit)

        def _clamp_potential(self, V: torch.Tensor) -> torch.Tensor:
            vmax = self._v_abs_max
            return torch.clamp(V, -vmax, vmax)

        def eval(self, p: Tuple[float, float, float]) -> float:
            P = torch.tensor(
                [[float(p[0]), float(p[1]), float(p[2])]],
                device=self._device,
                dtype=self._dtype,
            )
            P = self._clamp_points(P)
            with torch.no_grad():
                u = self.model(P)
                V = self.V_singular(P).unsqueeze(1) + u
                V = self._clamp_potential(V)
            return float(V.view(-1)[0].item())

        def eval_V_E_batched(
            self,
            P: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Batched potential/E-field evaluator.

            For compatibility with certify/energy helpers:
            - Returns V[N] as model + singular field.
            - Returns E[N,3] filled with NaNs (E-field not implemented).
            """
            P = P.to(device=self._device, dtype=self._dtype)
            P = self._clamp_points(P)
            with torch.no_grad():
                u = self.model(P)
                V = self.V_singular(P).unsqueeze(1) + u
                V = self._clamp_potential(V)
            V_out = V.view(-1)
            E = torch.full(
                (P.shape[0], 3),
                float("nan"),
                device=self._device,
                dtype=self._dtype,
            )
            return V_out, E

    solution = PINNSolution(
        model_=model,
        V_singular_fn=V_singular,
        device_=device,
        dtype_=base_dtype,
    )

    return {
        "solution": solution,
        "boundary_samples": boundary_samples,
        "final_train_pde": history["train_pde"][-1] if history["train_pde"] else float("nan"),
        "final_train_bc": history["train_bc"][-1] if history["train_bc"] else float("nan"),
        "training_history": history,
        "seed": seed,
        "bc_rmse": float(bc_rmse),
    }


# -----------------------------------------------------------------------------
# Back-compat stub
# -----------------------------------------------------------------------------


def pinn_train_eval_stub(
    spec: CanonicalSpec,
    cfg: PINNConfig,
    logger: JsonlLogger,
) -> Dict[str, Any]:
    """
    Simple deterministic stub for legacy call sites.

    Returns small fixed losses and a trivial boundary_samples array.
    """
    logger.info("Using pinn_train_eval_stub (no real PINN training).")
    return {
        "final_train_pde": 1e-4,
        "final_train_bc": 3e-6,
        "boundary_samples": [0.0] * 256,
        "bc_rmse": 0.0,
    }


# -----------------------------------------------------------------------------
# Standalone training step helper (for external loops)
# -----------------------------------------------------------------------------


def pinn_training_step(
    model: nn.Module,
    interior_sampler: "InteriorSampler",
    boundary_sampler: "BoundarySampler",
    V_singular_fn,
    pde_loss: nn.Module,
    bc_loss: nn.Module,
    n_collocation: int,
    n_boundary: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Single explicit training step, without owning the optimizer.

    This helper is intentionally minimal:
    - It builds batches using provided samplers,
    - Computes PDE and BC losses,
    - Returns (total_loss_tensor, {"pde": float, "bc": float}).

    The caller is responsible for:
    - zero_grad / backward / optimizer.step,
    - any AMP/autocast or GradScaler usage.
    """
    # Interior collocation samples
    X_int = interior_sampler.sample(
        n_collocation,
        device=device,
        dtype=dtype,
    ).requires_grad_(True)

    # Boundary samples (Dirichlet V=0)
    X_bc = boundary_sampler.sample(
        n_boundary,
        device=device,
        dtype=dtype,
    )
    V_bc_target = torch.zeros(X_bc.shape[0], device=device, dtype=dtype)

    # Network outputs
    u_int = model(X_int)
    u_bc = model(X_bc)
    V_bc_pred = V_singular_fn(X_bc).unsqueeze(1) + u_bc

    # Losses
    lp = pde_loss(u_int, X_int)
    lb = bc_loss(V_bc_pred.squeeze(), V_bc_target)
    total = lp + lb

    return total, {
        "pde": float(lp.item()),
        "bc": float(lb.item()),
    }
