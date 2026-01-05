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

| Model | Final Test Acc | Best Test Acc |
|-------|----------------|---------------|
| ResNet-32 | 71.19% | 71.24% |

### DML Training (2 Nodes)

| Node | Model | Final Test Acc | Best Test Acc |
|------|-------|----------------|---------------|
| 0 | ResNet-32 | 61.24% | 61.24% |
| 1 | ResNet-32 | 61.34% | 61.59% |

*Note: DML training is currently in progress, so results are preliminary.*

## TensorBoard Logs

```bash
tensorboard --logdir examples/CIFAR100/runs
```

Log locations:
- Independent training: `runs/pre-train/resnet32/`
- DML training: `runs/dml_2/{node_id}_resnet32/`
