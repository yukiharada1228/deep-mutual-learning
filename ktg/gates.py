import torch
import torch.nn as nn


class ThroughGate(nn.Module):
    def __init__(self, max_epoch):
        super(ThroughGate, self).__init__()

    def forward(self, loss, epoch, **kwargs):
        # Return the mean of per-sample losses
        return loss.mean()


class CutoffGate(nn.Module):
    def __init__(self, max_epoch):
        super(CutoffGate, self).__init__()

    def forward(self, loss, epoch, **kwargs):
        # Completely disable gradient contribution
        return torch.zeros_like(loss[0], requires_grad=True).sum()


class LinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(LinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch, **kwargs):
        # Linear schedule from 0 to 1 (normalized to reach 1 at final epoch)
        if self.max_epoch <= 1:
            loss_weight = float(epoch > 0)
        else:
            loss_weight = epoch / (self.max_epoch - 1)
        # Clip to account for numerical errors
        loss_weight = max(0.0, min(1.0, loss_weight))
        # Apply weight to per-sample losses and return mean
        return (loss * loss_weight).mean()


class CorrectGate(nn.Module):
    def __init__(self, max_epoch):
        super(CorrectGate, self).__init__()

    def forward(self, loss, epoch, student_logits, teacher_logits, label, **kwargs):
        # Filter samples based on teacher and student prediction correctness
        # as defined in the paper
        # Determine if student and teacher predictions are correct
        true_s = student_logits.argmax(dim=1) == label
        true_t = teacher_logits.argmax(dim=1) == label

        # Create masks for each case
        TT = ((true_t == 1) & (true_s == 1)).float()
        TF = ((true_t == 1) & (true_s == 0)).float()
        FT = ((true_t == 0) & (true_s == 1)).float()
        FF = ((true_t == 0) & (true_s == 0)).float()

        # Paper definition: TT=1, TF=1, FT=0, FF=0
        # Use only samples where teacher is correct
        mask = 1 * TT + 1 * TF + 0 * FT + 0 * FF

        # Apply mask and return mean
        return (loss * mask).mean()
