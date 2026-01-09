"""SimCLR Model with Projection Head.

Reference: https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """SimCLR model for self-supervised contrastive learning.

    Combines an encoder network with a projection head for contrastive learning.

    Args:
        encoder_func: Function that returns an encoder network.
        out_dim: Output dimension of projection head (default: 128).
    """

    def __init__(self, encoder_func, out_dim: int = 128) -> None:
        super(SimCLR, self).__init__()

        # Setup encoder network
        self.encoder = encoder_func()

        # Get encoder output dimension and remove classification head
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

        # Setup projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.input_dim, out_dim, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, _=None) -> list[torch.Tensor]:
        """Forward pass through encoder and projector for both views.

        Args:
            x1: First augmented view of shape (batch_size, channels, height, width).
            x2: Second augmented view of shape (batch_size, channels, height, width).
            _: Unused parameter (for API compatibility with labels).

        Returns:
            List of [z1, z2] where z1 and z2 are projected features
            for the two views, each of shape (batch_size, out_dim).
        """
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project to contrastive learning space
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return [z1, z2]
