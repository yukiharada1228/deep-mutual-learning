import torch
import torch.nn as nn


class ThroughGate(nn.Module):
    def __init__(self, max_epoch):
        super(ThroughGate, self).__init__()

    def forward(self, loss, epoch, **kwargs):
        return loss.mean()


class CutoffGate(nn.Module):
    def __init__(self, max_epoch):
        super(CutoffGate, self).__init__()

    def forward(self, loss, epoch, **kwargs):
        return loss.new_zeros((), requires_grad=True)


class LinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(LinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch, **kwargs):
        if self.max_epoch <= 1:
            loss_weight = float(epoch > 0)
        else:
            loss_weight = epoch / (self.max_epoch - 1)
        loss_weight = max(0.0, min(1.0, loss_weight))
        return (loss * loss_weight).mean()


class CorrectGate(nn.Module):
    def __init__(self, max_epoch):
        super(CorrectGate, self).__init__()

    def forward(self, loss, epoch, teacher_logits=None, label=None, **kwargs):
        if teacher_logits is None or label is None:
            return loss.mean()

        true_t = teacher_logits.argmax(dim=1) == label
        mask = true_t.float()

        return (loss * mask).sum() / (mask.sum() + 1e-8)
