# CIFAR-100 Experiments

Experiments on the CIFAR-100 dataset.

## Usage

### Independent Training
```bash
uv run independent_train.py --model resnet32
```

### DML Training (2 Nodes)
```bash
uv run dml_train.py --models resnet32 --num-nodes 2
```

## Training Configuration

- **Epochs**: 200
- **Batch size**: 64
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler**: CosineAnnealingLR

## Results

### Independent Training

| Model | Train Acc | Test Acc |
|-------|-----------|----------|
| ResNet-32 | 92.83% | 71.19% |

### DML Training

#### 2 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 88.52% | 72.04% |
| 1 | 88.92% | 72.19% |
| **Average** | **88.72%** | **72.12%** |

#### 3 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 89.56% | 72.52% |
| 1 | 89.70% | 73.31% |
| 2 | 88.77% | 72.10% |
| **Average** | **89.34%** | **72.64%** |

#### 4 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 90.28% | 73.11% |
| 1 | 90.00% | 72.99% |
| 2 | 89.42% | 73.63% |
| 3 | 89.87% | 73.10% |
| **Average** | **89.89%** | **73.21%** |

#### 5 Nodes (In Progress - 49/200 epochs)

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 53.78% | 45.84% |
| 1 | 54.21% | 49.91% |
| 2 | 54.21% | 48.69% |
| 3 | 54.49% | 48.38% |
| 4 | 54.30% | 46.75% |
| **Average** | **54.20%** | **47.91%** |

#### 6 Nodes (In Progress - 44/200 epochs)

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 53.59% | 48.74% |
| 1 | 52.86% | 50.12% |
| 2 | 53.26% | 47.99% |
| 3 | 53.35% | 47.58% |
| 4 | 53.12% | 47.92% |
| 5 | 52.94% | 48.54% |
| **Average** | **53.19%** | **48.48%** |

## TensorBoard Logs

```bash
tensorboard --logdir examples/CIFAR100/runs
```

Log locations:
- Independent training: `runs/pre-train/resnet32/`
- DML training: `runs/dml_2/{node_id}_resnet32/`
