from typing import Dict, Any, Tuple, Type

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    try:
        # Available in newer PyTorch; kept for compatibility, but not used
        # in the current PINNHarmonic implementation.
        from torch.utils.checkpoint import checkpoint_sequential as _ckpt_seq  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - older PyTorch
        _ckpt_seq = None
except Exception:  # pragma: no cover - checkpoint may be unavailable
    _ckpt = None
    _ckpt_seq = None

import numpy as np
import torch
import torch.nn as nn

from electrodrive.learn.encoding import ENCODING_DIM


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        scale: float,
    ):
        super().__init__()
        self.register_buffer(
            "B",
            torch.randn(
                input_dim,
                mapping_size,
            )
            * scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat(
            [
                torch.sin(x_proj),
                torch.cos(x_proj),
            ],
            dim=-1,
        )


class _ResidualBlock(nn.Module):
    """
    Simple MLP block: Linear + activation.

    Used inside PINNHarmonic so we can insert residual connections cleanly.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Fresh activation per block to avoid module sharing issues
        self.act = act_cls()

        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class _ResidualMLP(nn.Module):
    """
    MLP with optional residual connections every `residual_every` blocks.

    This is used as the internal trunk of PINNHarmonic.
    """

    def __init__(
        self,
        in_dim: int,
        width: int,
        depth: int,
        act_cls: Type[nn.Module],
        residual_every: int = 4,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth must be positive")
        if width <= 0:
            raise ValueError("width must be positive")

        self.in_dim = in_dim
        self.width = width
        self.depth = depth
        self.residual_every = max(int(residual_every), 1)

        blocks = []

        # First block: in_dim -> width (no residual into first)
        blocks.append(_ResidualBlock(in_dim, width, act_cls))

        # Subsequent hidden blocks: width -> width
        for _ in range(depth - 1):
            blocks.append(_ResidualBlock(width, width, act_cls))

        self.blocks = nn.ModuleList(blocks)

        # Output layer (kept separate; no residual)
        self.out = nn.Linear(width, 1)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        residual = None

        for idx, block in enumerate(self.blocks):
            # Start residual at the first width-preserving block of each group.
            # Do NOT start at idx == 0 (in_dim -> width).
            # For residual_every=4, starts at idx=1,5,9,...
            if idx > 0 and ((idx - 1) % self.residual_every) == 0:
                residual = h

            h = block(h)

            # Close group after exactly `residual_every` width-preserving blocks.
            # For residual_every=4, ends at idx=4,8,12,...
            if (
                (idx % self.residual_every) == 0
                and idx > 0
                and residual is not None
                and residual.shape == h.shape
            ):
                h = h + residual
                # Explicitly reset to avoid accidental carry-over if schedule changes
                residual = None

        # After the loop, ensure any final open residual group is closed.
        if residual is not None and residual.shape == h.shape:
            h = h + residual

        return self.out(h)


class PINNHarmonic(nn.Module):
    """Conditional PINN for harmonic potentials V(x; encoding)."""

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        spatial_dim = 3

        # Backwards-compatible hyperparameters.
        hidden_dim = int(
            config.get(
                "width",
                config.get("hidden_dim", 256),
            )
        )
        num_layers = int(
            config.get(
                "depth",
                config.get("num_layers", 4),
            )
        )
        activation = config.get("activation", "silu")
        use_fourier = config.get("use_fourier", True)
        fourier_scale = config.get("fourier_scale", 10.0)
        mapping_size = int(config.get("mapping_size", 64))

        if use_fourier:
            self.spatial_mapping = FourierFeatures(
                spatial_dim,
                mapping_size,
                fourier_scale,
            )
            in_dim = mapping_size * 2 + ENCODING_DIM
        else:
            self.spatial_mapping = None
            in_dim = spatial_dim + ENCODING_DIM

        # Residual / checkpointing controls.
        residual_every_raw = int(config.get("residual_every", 0))
        if residual_every_raw <= 1:
            self.residual_every = 0
        else:
            self.residual_every = residual_every_raw

        self.gradient_checkpointing = bool(
            config.get("gradient_checkpointing", False)
        )

        # Kept for compatibility; no longer used to drive checkpoint_sequential.
        self.checkpoint_segments = config.get("checkpoint_segments", None)

        if activation == "silu":
            act_cls: Type[nn.Module] = nn.SiLU
        elif activation == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Decide trunk implementation:
        # - If residuals are disabled AND gradient checkpointing is False:
        #     build the original nn.Sequential exactly to preserve legacy behavior.
        # - Otherwise:
        #     use _ResidualMLP with Xavier init and residuals.
        use_legacy_sequential = (
            self.residual_every == 0
            and not self.gradient_checkpointing
        )

        if num_layers < 2:
            use_legacy_sequential = True

        if use_legacy_sequential:
            layers = []
            cur = in_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(cur, hidden_dim))
                layers.append(act_cls())
                cur = hidden_dim
            layers.append(nn.Linear(cur, 1))
            self.network = nn.Sequential(*layers)
            self._use_residual_mlp = False
        else:
            self.mlp = _ResidualMLP(
                in_dim=in_dim,
                width=hidden_dim,
                depth=num_layers,
                act_cls=act_cls,
                residual_every=(
                    self.residual_every if self.residual_every > 0 else 4
                ),
            )
            self._use_residual_mlp = True

    def _forward_trunk(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through the trunk with optional gradient checkpointing.
        """
        # Legacy path: no residual MLP, no checkpointing.
        if not getattr(self, "_use_residual_mlp", False):
            return self.network(h)

        # Residual MLP path with optional non-reentrant checkpoint.
        use_ckpt = (
            self.gradient_checkpointing
            and _ckpt is not None
            and self.training
            and h.requires_grad
        )

        if not use_ckpt:
            return self.mlp(h)

        # Single non-reentrant checkpoint around the residual MLP.
        def _mlp_call(x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

        return _ckpt(
            _mlp_call,
            h,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        if self.spatial_mapping is not None:
            x_feat = self.spatial_mapping(x)
        else:
            x_feat = x

        if encoding.dim() == 1:
            encoding = encoding.unsqueeze(0).expand(
                x_feat.shape[0],
                -1,
            )

        h = torch.cat(
            [
                x_feat,
                encoding,
            ],
            dim=-1,
        )
        return self._forward_trunk(h)

    def compute_gradients_and_laplacian(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure we are in a gradient-enabled context for higher-order derivatives,
        # even if called under torch.no_grad() (e.g., during validation).
        with torch.enable_grad():
            x = (
                x.clone()
                .detach()
                .requires_grad_(True)
            )

            # Temporarily disable autocast to keep V connected to x in full precision.
            autocast_enabled = torch.is_autocast_enabled()
            if autocast_enabled:
                torch.set_autocast_enabled(False)

            try:
                V = self(x, encoding)
            finally:
                if autocast_enabled:
                    torch.set_autocast_enabled(True)

            if not V.requires_grad:
                raise RuntimeError(
                    "PINNHarmonic: V does not require grad; "
                    "ensure torch.enable_grad() is active in this context."
                )

            grad_V = torch.autograd.grad(
                V,
                x,
                grad_outputs=torch.ones_like(V),
                create_graph=True,
                retain_graph=True,
            )[0]

            lap = 0.0
            for i in range(x.shape[1]):
                d2 = torch.autograd.grad(
                    grad_V[:, i : i + 1],
                    x,
                    grad_outputs=torch.ones_like(
                        grad_V[:, i : i + 1]
                    ),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, i : i + 1]
                lap = lap + d2

        return V, lap

    def compute_loss(
        self,
        data: Dict[str, torch.Tensor],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        if (
            not data
            or "X" not in data
            or data["X"].numel() == 0
        ):
            device = next(self.parameters()).device
            return {
                "total": torch.tensor(
                    0.0,
                    device=device,
                    requires_grad=True,
                )
            }

        X = data["X"]
        E = data["encoding"]
        V_gt = data["V_gt"].unsqueeze(-1)
        is_boundary = data["is_boundary"]
        mask_finite = data["mask_finite"]

        V_pred, lap = self.compute_gradients_and_laplacian(X, E)
        losses: Dict[str, torch.Tensor] = {}

        w_bc = weights.get("bc_dirichlet", 1.0)
        if w_bc > 0:
            m = is_boundary & mask_finite
            if m.any():
                losses["bc_dirichlet"] = (
                    w_bc
                    * torch.mean(
                        (V_pred[m] - V_gt[m]) ** 2
                    )
                )

        w_pde = weights.get("pde_residual", 1.0)
        if w_pde > 0:
            m = (~is_boundary) & mask_finite
            if m.any():
                losses["pde_residual"] = (
                    w_pde
                    * torch.mean(
                        lap[m] ** 2
                    )
                )

        active = [
            v
            for v in losses.values()
            if torch.is_tensor(v)
        ]
        if active:
            total = sum(active)
        else:
            total = torch.tensor(
                0.0,
                device=X.device,
                requires_grad=True,
            )
        losses["total"] = total
        return losses