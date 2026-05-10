"""Contrastive loss functions for SimCLR and CRD.

Includes:
- NTXentLoss: NT-Xent (SimCLR) contrastive loss.
- CRDLoss: Contrastive relational distillation loss that aligns
  batch-wise all-view similarity structures between online peer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning (SimCLR).

    Accepts two calling conventions:

    - ``forward([z1, z2], label=None)`` — from a unified :class:`Edge` (supervision edge).
    - ``forward(z1, z2)`` — direct call.

    Args:
        batch_size:  Number of samples in a batch.
        temperature: Temperature parameter for scaling similarities (default: 0.5).

    Reference:
        Chen et al. "A Simple Framework for Contrastive Learning of
        Visual Representations." ICML 2020. https://arxiv.org/abs/2002.05709
    """

    def __init__(self, batch_size: int, temperature: float = 0.5) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.N = 2 * batch_size
        self.temperature = temperature
        self.mask = self._create_mask(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _create_mask(self, batch_size: int) -> torch.Tensor:
        mask = torch.ones((self.N, self.N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z1, z2=None) -> torch.Tensor:
        if z2 is None:
            if isinstance(z1, (list, tuple)):
                z1, z2 = z1
            else:
                raise ValueError(
                    "NTXentLoss requires either (z1, z2) or [z1, z2] as input."
                )

        z = torch.cat((z1, z2), dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.N, 1)
        mask = self.mask.to(sim.device)
        negative_samples = sim[mask].reshape(self.N, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        labels = torch.zeros(self.N, dtype=torch.long, device=z.device)
        return self.criterion(logits, labels) / self.N


class CRDLoss(nn.Module):
    """Contrastive relational distillation loss.

    This loss distills the pairwise similarity structure formed within a
    contrastive batch. Given two augmented views, it concatenates them into
    2N representations, computes all pairwise similarities except
    self-similarities, and aligns the student's similarity distribution with
    the teacher's via KL divergence.

    Unlike standard feature-level distillation, this loss does not directly
    force student embeddings to match teacher embeddings. Instead, it transfers
    the teacher's relational structure: how each view ranks and relates to all
    other views in the batch.

    This is inspired by DoGo-style online distillation, but differs from the
    official DoGo loss: it distills a SimCLR-style all-view similarity
    distribution of shape [2N, 2N-1], rather than a cross-view [N, N]
    similarity matrix.

    Args:
        temperature: Temperature for scaling similarity logits (default: 0.5).
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def _similarity_logits(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute masked pairwise similarity logits from two views.

        Returns:
            Tensor of shape [2N, 2N-1] after removing self-similarities.
        """
        batch_size = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # [2N, d]
        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        return sim[mask].view(2 * batch_size, -1)

    def forward(self, student_output, teacher_output) -> torch.Tensor:
        s_z1, s_z2 = student_output
        t_z1, t_z2 = teacher_output

        # Ensure that this edge only updates the student side.
        t_z1 = t_z1.detach()
        t_z2 = t_z2.detach()

        sim_s = self._similarity_logits(s_z1, s_z2)
        sim_t = self._similarity_logits(t_z1, t_z2)

        # Note on scaling:
        # We use reduction="batchmean", which divides the KL divergence
        # by the number of distributions, i.e. 2N.
        #
        # We additionally multiply by T^2, following the standard KD convention.
        # This keeps the gradient scale less sensitive to the chosen temperature.
        loss = F.kl_div(
            F.log_softmax(sim_s, dim=-1),
            F.softmax(sim_t, dim=-1),
            reduction="batchmean",
        ) * (self.temperature**2)

        return loss
