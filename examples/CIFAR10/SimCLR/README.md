# CIFAR-10 SimCLR Experiments

This directory contains experimental results for SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) on CIFAR-10.

## 1. Independent Training (Baseline)

Base models trained individually using the NT-Xent loss and LARS optimizer. Evaluation is performed using kNN (k=20).

```bash
uv run python independent_train.py --model resnet50
```

| Model | kNN Accuracy (Test) |
|-------|---------------------|
| ResNet50 | **91.06%** |