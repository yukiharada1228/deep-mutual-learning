import torch.nn as nn


class ThroughGate(nn.Module):
    def __init__(self, max_epoch):
        super(ThroughGate, self).__init__()

    def forward(self, loss, epoch):
        return loss


class CutoffGate(nn.Module):
    def __init__(self, max_epoch):
        super(CutoffGate, self).__init__()

    def forward(self, loss, epoch):
        # 勾配寄与を完全に無効化
        return loss.detach() * 0.0


class PositiveLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(PositiveLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        # 0 -> 1 の線形スケジュール（最終epochで1になるように正規化）
        if self.max_epoch <= 1:
            loss_weight = float(epoch > 0)
        else:
            loss_weight = epoch / (self.max_epoch - 1)
        return loss * loss_weight


class NegativeLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(NegativeLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        # 1 -> 0 の線形スケジュール（最終epochで0になるように正規化）
        if self.max_epoch <= 1:
            loss_weight = float(epoch == 0)
        else:
            loss_weight = (self.max_epoch - 1 - epoch) / (self.max_epoch - 1)
        # 数値誤差を考慮してクリップ
        loss_weight = max(0.0, min(1.0, loss_weight))
        return loss * loss_weight
