# CIFAR-100 Experiments

Experiments on the CIFAR-100 dataset.

## Key Findings

Deep Mutual Learning (DML) demonstrates consistent improvements over independent training on CIFAR-100:

- **Independent Training**: 71.19% test accuracy (92.83% train accuracy)
- **Best DML (4 nodes)**: 73.21% test accuracy (89.89% train accuracy)
- **Improvement**: +2.02% test accuracy

### Observations

1. **Generalization**: DML consistently achieves better test accuracy while maintaining lower training accuracy compared to independent training, indicating improved generalization and reduced overfitting.

2. **Optimal Configuration**: Performance peaks at 4 nodes (73.21%), with diminishing returns beyond this point. Configurations with 6 and 7 nodes still outperform independent training but show slightly lower accuracy than 4 nodes.

3. **Consistency**: All DML configurations (2-7 nodes) outperform independent training, demonstrating the robustness of the mutual learning approach.

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

#### 5 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 89.12% | 72.10% |
| 1 | 89.49% | 72.66% |
| 2 | 89.59% | 73.00% |
| 3 | 89.81% | 72.63% |
| 4 | 89.81% | 72.65% |
| **Average** | **89.56%** | **72.61%** |

#### 6 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 90.08% | 73.39% |
| 1 | 89.36% | 72.53% |
| 2 | 89.79% | 73.48% |
| 3 | 89.77% | 73.55% |
| 4 | 89.69% | 72.83% |
| 5 | 89.55% | 72.94% |
| **Average** | **89.71%** | **73.12%** |

#### 7 Nodes

| Node | Train Acc | Test Acc |
|------|-----------|----------|
| 0 | 89.81% | 73.06% |
| 1 | 89.70% | 73.19% |
| 2 | 89.14% | 72.75% |
| 3 | 89.76% | 73.19% |
| 4 | 89.41% | 72.73% |
| 5 | 89.28% | 72.67% |
| 6 | 90.00% | 72.93% |
| **Average** | **89.59%** | **72.93%** |

## TensorBoard Logs

```bash
tensorboard --logdir examples/CIFAR100/runs
```

Log locations:
- Independent training: `runs/pre-train/resnet32/`
- DML training: `runs/dml_2/{node_id}_resnet32/`
