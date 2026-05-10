"""Contrastive loss functions for SimCLR and DoGo.

Includes:
- NTXentLoss: NT-Xent (SimCLR) contrastive loss.
- DoGoLoss: "Distill on the Go" loss that aligns cross-view similarity
  distributions between online peer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning (SimCLR).

    This is the primary self-supervised objective that pulls positive pairs
    (augmented views of the same image) together and pushes negative pairs apart.

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


class DoGoLoss(nn.Module):
    """Distill on the Go (DoGo) loss.

    This loss distills the cross-view similarity structure formed within a
    contrastive batch. Given two augmented views (z1, z2), it computes the
    pairwise similarities between them [N, N] and aligns the student's
    distribution with the teacher's via KL divergence.

    Reference:
        Bhat et al. "Distill on the Go: Online knowledge distillation in
        self-supervised learning." CVPR Workshops 2021.
        https://arxiv.org/abs/2104.09866

    Technical Notes:
    - Uses nn.CosineSimilarity(dim=2) to match official implementation.
    - Matches distributions in a single direction (z1->z2).
    - Scales by T^2 to maintain gradient magnitude consistency.

    Args:
        temperature: Temperature for scaling similarity logits (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _similarity_logits(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute cross-view similarity logits between two batches.

        Returns:
            Tensor of shape [N, N].
        """
        # (N, 1, d) and (1, N, d) -> (N, N)
        return self.similarity_f(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature

    def forward(self, student_output, teacher_output) -> torch.Tensor:
        s_z1, s_z2 = student_output
        t_z1, t_z2 = teacher_output

        # Ensure that this edge only updates the student side.
        t_z1 = t_z1.detach()
        t_z2 = t_z2.detach()

        # Compute cross-view similarity logits [N, N]
        # We match how view1 relates to all samples in view2.
        sim_s = self._similarity_logits(s_z1, s_z2)
        sim_t = self._similarity_logits(t_z1, t_z2)

        # Distill: z1 -> z2 (row-wise softmax)
        log_p_s = F.log_softmax(sim_s, dim=1)
        p_t = F.softmax(sim_t, dim=1)

        # Apply KL divergence and T^2 scaling
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.temperature**2)
