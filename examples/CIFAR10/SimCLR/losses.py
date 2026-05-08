"""Contrastive loss functions for SimCLR and DisCO.

Includes:
- NTXentLoss:  NT-Xent (SimCLR) contrastive loss.
- DisCOLoss:   DisCO distillation loss for contrastive representations.
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


class DisCOLoss(nn.Module):
    """DisCO distillation loss for contrastive representations.

    Computes MSE between concatenated feature vectors of student and teacher.

    Accepts ``forward(student_pair, teacher_pair)`` where each is ``[z1, z2]``,
    as produced by a distillation :class:`Edge`.

    Reference:
        Gao et al. "DisCo: Remedy Self-supervised Learning on Lightweight
        Models with Distilled Contrastive Learning." ECCV 2022.
        https://arxiv.org/abs/2104.09124
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, student_output, teacher_output) -> torch.Tensor:
        s_z1, s_z2 = student_output
        t_z1, t_z2 = teacher_output
        fvec_s = torch.cat((s_z1, s_z2), dim=0)
        fvec_t = torch.cat((t_z1.detach(), t_z2.detach()), dim=0)
        return self.criterion(fvec_s, fvec_t)
