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

    def forward(self, target_outputs: tuple, labels=None) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            target_outputs: Tuple of (z1, z2) where z1 and z2 are projected
                representations from two augmented views.
            labels: Ignored. Present for API compatibility with supervised losses.

        Returns:
            Scalar loss value.
        """
        # Extract embeddings from both views (via edge self-loop in graph)
        z1, z2 = target_outputs[0], target_outputs[1]

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


class DoGoLoss(nn.Module):
    """DoGo (Distill on the Go) Loss for Online Knowledge Distillation.

    Implements online knowledge distillation loss using KL divergence between
    similarity distributions from two peer models. Each model learns from
    self-supervised contrastive learning and mutual distillation.

    Reference:
        Bhat et al. "Distill on the Go: Online knowledge distillation in
        self-supervised learning." CVPR 2021 Workshop.
        https://arxiv.org/abs/2104.09866

    Args:
        temperature: Temperature parameter for softmax scaling (default: 0.1).
    """

    def __init__(self, temperature: float = 1.0, loss_weight: float = 1.0) -> None:
        super(DoGoLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        # Cosine similarity function (computes similarity along dim=2)
        self.similarity_fn = nn.CosineSimilarity(dim=2)
        # KL divergence loss with batch mean reduction
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def _compute_similarity_matrix(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity matrix.

        Args:
            z1: First set of embeddings, shape (batch_size, dim).
            z2: Second set of embeddings, shape (batch_size, dim).

        Returns:
            Similarity matrix of shape (batch_size, batch_size).
        """
        # Expand dimensions to compute all pairwise similarities
        # z1.unsqueeze(1): (batch_size, 1, dim)
        # z2.unsqueeze(0): (1, batch_size, dim)
        # Result: (batch_size, batch_size)
        return self.similarity_fn(z1.unsqueeze(1), z2.unsqueeze(0))

    def forward(self, target_outputs: tuple, source_outputs: tuple) -> torch.Tensor:
        """Compute DoGo distillation loss.

        Args:
            target_outputs: Tuple of (z1_target, z2_target) from target model.
            source_outputs: Tuple of (z1_source, z2_source) from source model.

        Returns:
            KL divergence loss between target and source similarity distributions.
        """
        # Extract embeddings from target model (student)
        z1_target, z2_target = target_outputs[0], target_outputs[1]
        # Extract embeddings from source model (teacher)
        z1_source, z2_source = source_outputs[0], source_outputs[1]

        # Compute similarity matrices
        sim_target = self._compute_similarity_matrix(z1_target, z2_target)
        sim_source = self._compute_similarity_matrix(z1_source, z2_source)

        # Compute KL divergence loss between distributions
        # Target model: log probabilities (student)
        log_prob_target = F.log_softmax(sim_target / self.temperature, dim=-1)
        # Source model: probabilities (teacher, detached from gradient)
        prob_source = F.softmax(sim_source.detach() / self.temperature, dim=-1)

        loss = self.criterion(log_prob_target, prob_source) * (self.temperature**2)
        return loss
