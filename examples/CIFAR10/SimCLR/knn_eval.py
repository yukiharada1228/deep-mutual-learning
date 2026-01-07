"""K-Nearest Neighbors (KNN) evaluation for self-supervised learning.

Implementation based on DINO:
https://github.com/facebookresearch/dino/blob/main/eval_knn.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple:
    """Extract features from a model for all samples in the data loader.

    Args:
        model: Neural network model (encoder).
        data_loader: DataLoader containing the dataset.
        device: Device to run the model on.

    Returns:
        Tuple of (features, labels) where features are L2-normalized.
    """
    model.eval()
    features_list = []
    labels_list = []

    for images, labels in data_loader:
        images = images.to(device)

        # Extract features (use encoder output)
        if hasattr(model, "encoder"):
            # For SimCLR model
            features = model.encoder(images)
        else:
            # For standard models
            features = model(images)

        # Normalize features
        features = nn.functional.normalize(features, dim=1, p=2)

        features_list.append(features.cpu())
        labels_list.append(labels)

    # Concatenate all batches
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels


@torch.no_grad()
def knn_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
) -> dict:
    """Perform KNN classification with temperature-weighted voting.

    Args:
        train_features: Training feature vectors (N_train x D), L2-normalized.
        train_labels: Training labels (N_train,).
        test_features: Test feature vectors (N_test x D), L2-normalized.
        test_labels: Test labels (N_test,).
        k: Number of nearest neighbors to consider.
        temperature: Temperature for softmax weighting (lower = sharper).
        num_classes: Number of classes in the dataset.

    Returns:
        Dictionary containing top1 and top5 accuracies.
    """
    train_features = train_features.t()  # Transpose for efficient dot product
    num_test_images = test_features.shape[0]
    num_chunks = num_test_images // 100 + (1 if num_test_images % 100 > 0 else 0)

    top1_correct = 0
    top5_correct = 0
    total = 0

    # Process test images in chunks to save memory
    for chunk_idx in range(num_chunks):
        # Get chunk boundaries
        start_idx = chunk_idx * 100
        end_idx = min((chunk_idx + 1) * 100, num_test_images)
        features = test_features[start_idx:end_idx]
        targets = test_labels[start_idx:end_idx]
        batch_size = features.shape[0]

        # Compute similarity: dot product (features are already normalized)
        similarity = torch.mm(features, train_features)

        # Get top-k nearest neighbors
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        # Retrieve labels of k nearest neighbors
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        # Temperature-weighted voting
        # Apply temperature scaling: exp(similarity / T)
        distances_transform = distances.clone().div_(temperature).exp_()

        # Create one-hot encoding for retrieved labels
        retrieval_one_hot = torch.zeros(
            batch_size * k, num_classes, device=train_features.device
        )
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        retrieval_one_hot = retrieval_one_hot.view(batch_size, k, num_classes)

        # Weight votes by similarity scores
        distances_transform = distances_transform.unsqueeze(-1)
        probs = (retrieval_one_hot * distances_transform).sum(1)

        # Get predictions
        _, predictions = probs.topk(5, largest=True, sorted=True)

        # Compute accuracy
        targets_expanded = targets.view(-1, 1)
        top1_correct += (predictions[:, :1] == targets_expanded).sum().item()
        top5_correct += (predictions == targets_expanded).sum().item()
        total += batch_size

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    return {"top1": top1_acc, "top5": top5_acc}


@torch.no_grad()
def evaluate_knn(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
) -> dict:
    """Evaluate a model using KNN classification.

    This is a convenient wrapper that extracts features and performs KNN classification.

    Args:
        model: Neural network model to evaluate.
        train_loader: DataLoader for training set (used as KNN database).
        test_loader: DataLoader for test set.
        device: Device to run the model on.
        k: Number of nearest neighbors.
        temperature: Temperature for softmax weighting.
        num_classes: Number of classes in the dataset.

    Returns:
        Dictionary containing top1 and top5 accuracies.
    """
    # Extract features for train and test sets
    print("  Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)

    print("  Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # Move to GPU for KNN computation if available
    if device.type == "cuda" or device.type == "mps":
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)

    # Perform KNN classification
    print("  Running KNN classification...")
    results = knn_classifier(
        train_features,
        train_labels,
        test_features,
        test_labels,
        k=k,
        temperature=temperature,
        num_classes=num_classes,
    )

    return results
