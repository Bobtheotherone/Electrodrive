# electrodrive/learn/models/moi_symbolic.py
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from electrodrive.utils.config import K_E
from electrodrive.learn.encoding import ENCODING_DIM


class MoISymbolic(nn.Module):
    """Generalized symbolic Method-of-Images inducer.

    Learns sparse image systems conditioned on encoded problem specs.
    """

    def __init__(
        self, config: Dict[str, Any]
    ):
        super().__init__()
        self.max_images = config.get(
            "max_images", 8
        )
        hidden_dim = config.get(
            "hidden_dim", 256
        )
        input_dim = ENCODING_DIM
        output_dim = (
            5 * self.max_images
        )  # (q, x, y, z, logit)

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        encoding: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            encoding.dim()
            == 1
        ):
            encoding = (
                encoding.unsqueeze(0)
            )
        out = self.network(
            encoding
        )
        B = out.shape[0]
        out = out.view(
            B,
            self.max_images,
            5,
        )
        mags = out[
            ..., 0
        ]
        locs = out[
            ..., 1:4
        ]
        conf = torch.sigmoid(
            out[
                ..., 4
            ]
        )
        return mags, locs, conf

    def evaluate_potential(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        if (
            x.dim() != 2
            or x.shape[1] != 3
        ):
            raise ValueError(
                "x must be [N, 3]"
            )
        mags, locs, conf = self(
            encoding
        )
        mags_b = mags[
            0
        ].unsqueeze(0).expand(
            x.shape[0],
            -1,
        )
        locs_b = locs[
            0
        ].unsqueeze(0).expand(
            x.shape[0],
            -1,
            -1,
        )
        conf_b = conf[
            0
        ].unsqueeze(0).expand(
            x.shape[0],
            -1,
        )
        R_vec = (
            x.unsqueeze(1)
            - locs_b
        )
        R = torch.linalg.norm(
            R_vec,
            dim=2,
        ).clamp_min(1e-12)
        V = K_E * torch.sum(
            (mags_b * conf_b)
            / R,
            dim=1,
        )
        return V.unsqueeze(-1)

    def compute_loss(
        self,
        data: Dict[
            str,
            torch.Tensor,
        ],
        weights: Dict[
            str,
            float,
        ],
    ) -> Dict[
        str, torch.Tensor
    ]:
        if (
            not data
            or "X"
            not in data
            or data["X"].numel()
            == 0
        ):
            device = next(
                self.parameters()
            ).device
            return {
                "total": torch.tensor(
                    0.0,
                    device=device,
                    requires_grad=True,
                )
            }

        X = data["X"]
        E = data["encoding"]
        V_gt = data[
            "V_gt"
        ].unsqueeze(-1)
        mask_finite = (
            data[
                "mask_finite"
            ]
        )

        V_pred = self.evaluate_potential(
            X, E
        )
        losses: Dict[
            str, torch.Tensor
        ] = {}

        w_sup = weights.get(
            "bc_dirichlet",
            1.0,
        )
        if (
            w_sup > 0
            and mask_finite.any()
        ):
            losses[
                "supervised"
            ] = (
                w_sup
                * torch.mean(
                    (
                        V_pred[
                            mask_finite
                        ]
                        - V_gt[
                            mask_finite
                        ]
                    )
                    ** 2
                )
            )

        w_sp = weights.get(
            "sparsity",
            1e-4,
        )
        if w_sp > 0:
            _, _, conf = self(E)
            losses[
                "sparsity"
            ] = (
                w_sp
                * conf.mean()
            )

        active = [
            v
            for v in losses.values()
            if torch.is_tensor(v)
        ]
        if active:
            total = sum(
                active
            )
        else:
            total = (
                torch.tensor(
                    0.0,
                    device=X.device,
                    requires_grad=True,
                )
            )
        losses[
            "total"
        ] = total
        return losses