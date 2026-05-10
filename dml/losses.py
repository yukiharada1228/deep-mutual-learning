import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    """Standard Knowledge Distillation loss using KL divergence.

    Aligns the student's log-probabilities with the teacher's probabilities,
    scaled by temperature T.  Includes the T^2 multiplier to maintain gradient
    magnitude consistent with standard cross-entropy.

    Args:
        temperature: Softmax temperature (default: 1.0).
        reduction:   Reduction method for KLDivLoss (default: "batchmean").
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        T = self.temperature
        p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        return self.kl_div(p_s, p_t) * (T**2)
