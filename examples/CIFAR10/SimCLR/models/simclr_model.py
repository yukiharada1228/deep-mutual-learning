"""SimCLR Model with Projection Head.

Reference: https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn


class SimCLRProjector(nn.Module):
    """Projection head (MLP) for SimCLR.

    Maps encoder representations to a lower-dimensional space where
    contrastive loss is applied.

    Args:
        input_dim: Dimension of input features from encoder.
        out_dim: Dimension of output projections (default: 128).
    """

    def __init__(self, input_dim: int, out_dim: int = 128) -> None:
        super(SimCLRProjector, self).__init__()
        self.out_dim = out_dim
        # 2-layer MLP: input_dim -> input_dim -> out_dim
        # Following SimCLR Appendix B.9 for CIFAR
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to contrastive learning space.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Projected features of shape (batch_size, out_dim).
        """
        z = self.projector(x)
        return z


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

    @torch.no_grad()
    def encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features without gradient computation.

        Args:
            x: Input images of shape (batch_size, channels, height, width).

        Returns:
            Encoder features of shape (batch_size, input_dim).
        """
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract projected features without gradient computation.

        Args:
            x: Input images of shape (batch_size, channels, height, width).

        Returns:
            Projected features of shape (batch_size, out_dim).
        """
        return self.projector(self.encoder(x))

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, _=None
    ) -> list[torch.Tensor]:
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
