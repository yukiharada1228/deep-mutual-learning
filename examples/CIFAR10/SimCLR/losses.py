"""Contrastive loss functions for SimCLR and DoGo.

Includes:
- NTXentLoss:  NT-Xent (SimCLR) contrastive loss.
- DoGoLoss:    DoGo online mutual distillation loss for self-supervised learning.
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
        if z2 is None and isinstance(z1, (list, tuple)):
            z1, z2 = z1

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
    """DoGo online mutual distillation loss for self-supervised learning.

    Aligns each model's softmax probability distribution over pairwise similarity
    scores with that of the peer model, via KL divergence.  Both models are trained
    simultaneously without a pre-trained teacher (online / mutual distillation).

    Accepts ``forward(student_pair, teacher_pair)`` where each argument is
    ``[z1, z2]`` as produced by a distillation :class:`Edge`.  The ``teacher_pair``
    tensors are expected to already be detached (the :class:`Edge` handles this).

    Args:
        temperature: Temperature for scaling similarity logits. The official
            implementation uses 0.1 for distillation, distinct from the
            contrastive loss temperature (typically 0.5).

    Reference:
        Bhat et al. "Distill on the Go: Online knowledge distillation in
        self-supervised learning." CVPR Workshops 2021.
        https://arxiv.org/abs/2104.09866
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def _similarity_distribution(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked softmax similarity distribution from two views.

        Returns:
            Tensor of shape [2N, 2N-1] after removing self-similarities.
        """
        N = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # [2N, d]
        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # Mask out diagonal (self-similarity) → [2N, 2N-1]
        mask = ~torch.eye(2 * N, dtype=torch.bool, device=sim.device)
        return sim[mask].view(2 * N, -1)

    def forward(self, student_output, teacher_output) -> torch.Tensor:
        s_z1, s_z2 = student_output
        t_z1, t_z2 = teacher_output  # expected to be detached by Edge

        sim_s = self._similarity_distribution(s_z1, s_z2)
        sim_t = self._similarity_distribution(t_z1, t_z2)

        # Note on scaling:
        # We use reduction="batchmean", which divides the KL divergence
        # by the number of distributions, i.e. 2N.
        #
        # We also multiply by temperature squared (T^2) to maintain the
        # magnitude of gradients, consistent with standard KD.
        #
        # Compared to the official implementation (which uses "mean" reduction
        # and no T^2 scaling), our loss with weight=1.0 is naturally on a
        # similar scale to their weight=100.0.
        loss = F.kl_div(
            F.log_softmax(sim_s, dim=-1),
            F.softmax(sim_t, dim=-1),
            reduction="batchmean",
        ) * (self.temperature**2)
        return loss
