# CIFAR-100 Experiments

This directory contains experimental results for CIFAR-100 classification using various training strategies.

## 1. Independent Training (Baseline)

Base models trained individually without any knowledge transfer.

```
python independent_train.py --model resnet32
python independent_train.py --model resnet110
python independent_train.py --model wideresnet28_2
```

| Model | Test Accuracy |
|-------|--------------|
| ResNet32 | **71.54%** |
| ResNet110 | **73.65%** |
| WideResNet28-2 | **75.43%** |

## 2. Knowledge Distillation (KD)

Standard Knowledge Distillation (Hinton et al.) with Temperature $T=2.0$.

```
python kd_train.py --teachers wideresnet28_2 --students resnet32 --temperature 2.0
```

- **Teacher**: WideResNet28-2 (pre-trained, frozen)
- **Student**: ResNet32

| Model | Test Accuracy |
|-------|--------------|
| ResNet32 (Student) | **73.53%** |

## 3. Deep Mutual Learning (DML)

### 3.1 DML with 2 Nodes (T=1.0)

Collaborative learning between two models with Temperature $T=1.0$.

```
python dml_train.py --models resnet32 --num-nodes 2 --temperature 1.0
```

- **Node 0**: ResNet32
- **Node 1**: ResNet32

| Model | Test Accuracy |
|-------|--------------|
| Node 0 (ResNet32) | **72.32%** |
| Node 1 (ResNet32) | **72.47%** |
