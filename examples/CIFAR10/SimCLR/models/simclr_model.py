"""SimCLR Model with Projection Head.

Reference: https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """SimCLR model for self-supervised contrastive learning.

    Args:
        encoder_func: Function that returns an encoder network.
        out_dim: Output dimension of projection head (default: 128).
        num_proj_layers: Number of projection head layers (default: 3).
    """

    def __init__(
        self, encoder_func, out_dim: int = 128, num_proj_layers: int = 2
    ) -> None:
        super(SimCLR, self).__init__()

        self.encoder = encoder_func()

        if hasattr(self.encoder, "fc"):
            self.input_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, "linear"):
            self.input_dim = self.encoder.linear.in_features
            self.encoder.linear = nn.Identity()
        else:
            raise ValueError(
                "Encoder must have 'fc' or 'linear' attribute for output dimension"
            )

        # Projection head matching the official SimCLR implementation:
        # middle layers: Linear(d,d) + BN + ReLU
        # final layer:   Linear(d,out_dim, bias=False) + BN
        layers = []
        for j in range(num_proj_layers):
            is_last = j == num_proj_layers - 1
            out = out_dim if is_last else self.input_dim
            layers.append(nn.Linear(self.input_dim, out, bias=not is_last))
            layers.append(nn.BatchNorm1d(out))
            if not is_last:
                layers.append(nn.ReLU())
        self.projector = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, _=None) -> list[torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return [z1, z2]
