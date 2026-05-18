# SimCLR Examples on CIFAR-10

This directory contains implementations of SimCLR (Self-supervised Learning) and its extension with the Distill on the Go (DoGo) paradigm.

## 1. Independent Training (Standard SimCLR)

Train one or more models independently using the standard NT-Xent loss.

```bash
uv run python independent_train.py --model resnet50 --batch-size 512
```

| Model | kNN Accuracy (Test) |
|-------|---------------------|
| ResNet50 | **91.06%** |

## 2. DoGo Training (Online Mutual Distillation)

Two or more models learn collaboratively by aligning their batch-wise cross-view similarity distributions. This is a form of online mutual distillation without a pre-trained teacher.

```bash
uv run python dogo_train.py --models resnet50 --batch-size 256
```

Unlike standard feature-level distillation, DoGo aligns the **cross-view similarity distributions** between the augmented views in each batch. This allows models to share their "view of the world" (how each image in the batch relates to the others) without forcing their absolute feature coordinates to match.

| Model | kNN Accuracy (Test) |
|-------|---------------------|
| Node 0 (ResNet50) | **91.61%%** |
| Node 1 (ResNet50) | **91.91%%** |

## Utilities

- `models/`: ResNet implementations optimized for CIFAR.
- `training_utils.py`: Shared utilities for data loading, optimization, and SimCLR model creation.
- `losses.py`: Implementation of `NTXentLoss` and `DoGoLoss`.
- `knn_eval.py`: kNN-based evaluation for self-supervised features.
