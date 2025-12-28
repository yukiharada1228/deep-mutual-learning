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

    def forward(self, loss, epoch, teacher_logits=None, label=None, **kwargs):
        # Filter samples based on teacher prediction correctness
        # as defined in the paper

        # If required arguments are not provided (e.g., self-edge case),
        # behave like ThroughGate and return mean of all samples
        if teacher_logits is None or label is None:
            return loss.mean()

        # Determine if teacher predictions are correct
        # Compare teacher's predicted class (argmax) with ground truth label
        # Result: boolean tensor where True = teacher correct, False = teacher wrong
        true_t = teacher_logits.argmax(dim=1) == label

        # Paper definition: Use only samples where teacher is correct
        # TT (teacher correct, student correct) = 1
        # TF (teacher correct, student wrong) = 1
        # FT (teacher wrong, student correct) = 0
        # FF (teacher wrong, student wrong) = 0
        # Therefore, mask = teacher correctness
        # Convert boolean to float: True -> 1.0, False -> 0.0
        mask = true_t.float()

        # Apply mask and compute mean over valid samples only
        # Add epsilon to avoid division by zero while preserving gradients
        return (loss * mask).sum() / (mask.sum() + 1e-8)
