"""Classification evaluation utilities."""

import torch

from dml.utils import AverageMeter


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int,] = (1,)
) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100 * correct_k / batch_size)
    return res


def create_classification_evaluator(val_dataloader, score_fn):
    """Create a per-node eval_fn for classification tasks.

    Returns a callable ``(model, device) → float`` compatible with
    :attr:`Node.eval_fn <dml.Node.eval_fn>`.

    Args:
        val_dataloader: Validation DataLoader yielding ``(image, label)`` tuples.
        score_fn:       ``(output, label) → float`` scoring function.
    """

    def evaluate(model, device):
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
