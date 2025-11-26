import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivLoss, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_pred, y_gt):
        y_pred_soft = self.softmax(y_pred / self.T)
        y_gt_soft = self.softmax(y_gt.detach() / self.T)
        # 温度スケーリングに伴う勾配スケール補正 T^2
        return (self.T**2) * self.kl_divergence(y_pred_soft, y_gt_soft)

    def kl_divergence(self, student, teacher):
        kl = teacher * torch.log((teacher / (student + 1e-10)) + 1e-10)
        kl = kl.sum(dim=1)
        loss = kl.mean()
        return loss
