"""Loss functions for Self-Supervised Contrastive Learning.

Includes:
- SimCLR NT-Xent Loss: https://arxiv.org/abs/2002.05709
- DoGo Distillation Loss: https://arxiv.org/abs/2104.09866
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR contrastive learning.

    Args:
        batch_size: Number of samples in a batch.
        temperature: Temperature parameter for scaling similarities (default: 0.5).
    """

    def __init__(self, batch_size: int, temperature: float = 0.5) -> None:
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        # Total number of samples (2 augmented views per sample)
        self.N = 2 * batch_size
        # Temperature parameter for NT-Xent loss
        self.temperature = temperature
        # Mask to extract positive/negative pair similarities
        self.mask = self._create_correlated_mask(batch_size)
        # Cross-entropy loss with softmax
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _create_correlated_mask(self, batch_size: int) -> torch.Tensor:
        """Create mask to filter out self-similarities and positive pairs.

        Args:
            batch_size: Number of samples in a batch.

        Returns:
            Boolean mask of shape (2*batch_size, 2*batch_size).
        """
        mask = torch.ones((self.N, self.N), dtype=bool)
        # Set diagonal to False (remove self-similarities)
        mask.fill_diagonal_(False)

        # Set positive pairs to False
        for i in range(batch_size):
            # Remove positive pair similarities at positions (i, batch_size+i) and vice versa
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False

        return mask

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: Projected representations from view 1, shape (batch_size, out_dim).
            z2: Projected representations from view 2, shape (batch_size, out_dim).

        Returns:
            Scalar loss value.
        """

        # Compute similarities between all samples
        # Concatenate outputs from both views into a single tensor
        z = torch.cat((z1, z2), dim=0)
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        # Compute cosine similarity matrix for all pairs and scale by temperature
        sim = torch.matmul(z, z.T) / self.temperature

        # Extract positive and negative pair similarities
        # Extract positive pair similarities (i->j)
        sim_i_j = torch.diag(sim, self.batch_size)
        # Extract positive pair similarities (j->i)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # Combine positive pairs into a single tensor
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.N, 1)
        # Extract negative pair similarities only
        # Move mask to same device as similarity matrix
        mask = self.mask.to(sim.device)
        negative_samples = sim[mask].reshape(self.N, -1)
        # Concatenate positive and negative pairs
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Compute NT-Xent loss
        # Create labels: positive pair similarity is at position 0
        labels = torch.zeros(self.N, dtype=torch.long, device=z.device)
        # Compute loss (sum) and normalize by number of samples (average)
        loss = self.criterion(logits, labels) / self.N

        return loss


class DisCOLoss(nn.Module):
    """DisCO (Distilled Contrastive Learning) Loss for Knowledge Distillation.

    Implements knowledge distillation using MSE loss between feature vectors
    from two peer models. The loss measures the difference in features for
    the same samples between target and source models.

    Reference:
        Gao et al. "DisCo: Remedy Self-supervised Learning on Lightweight
        Models with Distilled Contrastive Learning." ECCV 2022.
        https://arxiv.org/abs/2104.09124
    """

    def __init__(self):
        super(DisCOLoss, self).__init__()
        # Loss function
        self.criterion = nn.MSELoss()

    def forward(
        self,
        z1_student: torch.Tensor,
        z2_student: torch.Tensor,
        z1_teacher: torch.Tensor,
        z2_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DisCO distillation loss.

        Args:
            z1_student: Projected features from student view 1.
            z2_student: Projected features from student view 2.
            z1_teacher: Projected features from teacher view 1.
            z2_teacher: Projected features from teacher view 2.

        Returns:
            MSE loss between student and teacher feature vectors.
        """
        z1_m1 = z1_student
        z2_m1 = z2_student
        z1_m2 = z1_teacher.detach()
        z2_m2 = z2_teacher.detach()

        # Knowledge transfer based on DisCO: difference in features for same samples
        # Concatenate outputs -> [batch*2, dimension]
        fvec_m1 = torch.cat((z1_m1, z2_m1), dim=0)
        fvec_m2 = torch.cat((z1_m2, z2_m2), dim=0)
        # Compute loss
        loss = self.criterion(fvec_m1, fvec_m2.detach())
        return loss
