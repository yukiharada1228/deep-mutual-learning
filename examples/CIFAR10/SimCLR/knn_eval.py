"""KNN evaluation for self-supervised representation quality.

Reference: https://github.com/facebookresearch/dino/blob/main/eval_knn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def _extract_features(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features_list, labels_list = [], []
    for images, labels in data_loader:
        images = images.to(device)
        features = model.encoder(images) if hasattr(model, "encoder") else model(images)
        features_list.append(F.normalize(features, dim=1, p=2).cpu())
        labels_list.append(labels)
    return torch.cat(features_list), torch.cat(labels_list)


@torch.no_grad()
def _knn_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int,
    temperature: float,
    num_classes: int,
) -> dict:
    train_features = train_features.t()
    num_test = test_features.shape[0]
    top1_correct = top5_correct = total = 0

    for start in range(0, num_test, 100):
        features = test_features[start : start + 100]
        targets = test_labels[start : start + 100]
        batch_size = features.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved = torch.gather(candidates, 1, indices)

        weights = distances.clone().div_(temperature).exp_()
        one_hot = torch.zeros(batch_size * k, num_classes, device=train_features.device)
        one_hot.scatter_(1, retrieved.view(-1, 1), 1)
        probs = (one_hot.view(batch_size, k, num_classes) * weights.unsqueeze(-1)).sum(1)

        _, predictions = probs.topk(5, largest=True, sorted=True)
        targets_exp = targets.view(-1, 1)
        top1_correct += (predictions[:, :1] == targets_exp).sum().item()
        top5_correct += (predictions == targets_exp).sum().item()
        total += batch_size

    return {"top1": 100.0 * top1_correct / total, "top5": 100.0 * top5_correct / total}


def evaluate_knn(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
) -> dict:
    """Evaluate a model using KNN classification on extracted features.

    Args:
        model: SimCLR model with an ``encoder`` attribute.
        train_loader: DataLoader for KNN database (training set).
        test_loader: DataLoader for query set (test set).
        device: Device to run feature extraction on.
        k: Number of nearest neighbours.
        temperature: Softmax temperature for weighted voting.
        num_classes: Number of target classes.

    Returns:
        Dict with ``top1`` and ``top5`` accuracy (%).
    """
    train_features, train_labels = _extract_features(model, train_loader, device)
    test_features, test_labels = _extract_features(model, test_loader, device)

    if device.type in ("cuda", "mps"):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)

    return _knn_classifier(
        train_features, train_labels, test_features, test_labels,
        k, temperature, num_classes,
    )
