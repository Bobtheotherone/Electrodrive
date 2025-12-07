import torch
import torch.nn as nn
import numpy as np


class FourierMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8, fourier_features=True, fourier_scale=10.0):
        super().__init__()
        self.fourier_features = fourier_features
        if fourier_features:
            # use hidden_dim features total (sin+cos halves)
            ff_dim = hidden_dim // 2
            self.register_buffer("B", torch.randn(input_dim, ff_dim) * fourier_scale)
            enc_dim = hidden_dim
        else:
            enc_dim = input_dim

        layers = [nn.Linear(enc_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_features:
            x_proj = 2.0 * np.pi * x @ self.B
            x_enc = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        else:
            x_enc = x
        return self.net(x_enc)




