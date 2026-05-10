"""Evaluation utilities for classification tasks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.eval import AverageMeter


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k.

    Args:
        output: Logits tensor of shape [B, C].
        target: Target tensor of shape [B].
        topk:   Tuple of k values for which to calculate accuracy.

    Returns:
        List of tensors representing accuracy @ k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(100 * correct_k / batch_size)
    return res


def create_classification_evaluator(val_dataloader: DataLoader, score_fn: callable):
    """Create a per-node evaluation function for classification.

    Returns a callable ``(model, device) → float`` compatible with
    :attr:`Node.eval_fn <dml.Node.eval_fn>`.

    Args:
        val_dataloader: DataLoader yielding ``(image, label)`` tuples.
        score_fn:       Callable ``(logits, label) → float`` (e.g. accuracy).
    """

    def evaluate(model: nn.Module, device: torch.device) -> float:
        model.eval()
        meter = AverageMeter()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                meter.update(score_fn(output, labels), labels.size(0))
        return meter.avg

    return evaluate
