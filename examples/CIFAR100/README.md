## CIFAR-100 Experiment Notes (KTG)

This directory contains the complete set of experiments for KTG (Knowledge Transfer Graph) on CIFAR-100. Use `dcl_train.py` for the search, `evaluation/dcl_test.py` and `evaluation/dml_test.py` for retraining and evaluation, and `pre_train.py` for pre-training single models.

### Highlights (TL;DR) — Focus on Node0 `resnet32`
- **pre-train (test)**: 71.44%
- **DML (test)**: 73.55% (+2.11pt vs pre-train)
- **DCL (test)**: 73.64% (+2.20pt vs pre-train)

### Table of Contents
- Dataset and preprocessing
- Common training settings
- Available models
- DCL search summary and configuration
- Retraining + test results (DCL / DML)
- Single model test (pre-train)
- Quick Start
- Notes

---

### Dataset and preprocessing
- **Split**: fixed `train=40,000 / val=10,000`
- **Test operation**: with `use_test_mode=True`, train on `train+val` and evaluate on `test` (normalization statistics are also computed from `train+val`)
- **Preprocessing**:
  - Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
  - Eval: ToTensor + Normalize

### Common training settings
- Batch size: 64
- Epochs: 200 (CosineAnnealingLR, `eta_min=0`)
- Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
- Num classes: 100

### Available models
- `resnet32`, `resnet110`, `wideresnet28_2`

### DCL search summary
- Study: `dcl_3` (number of nodes = 3)
- Log: `examples/CIFAR-100/optuna/dcl_3/optuna.log`
- Best trial: **Trial 17**
- Best score: **Top-1 72.21%** (estimated on `val`)
- Trial outputs: `examples/CIFAR-100/runs/dcl_3/0017/`
- Number of trials: 100

#### Best trial configuration (Trial 17)
- Models (node0→2): `[resnet32, resnet32, wideresnet28_2]`
- Gate matrix (row = i node = receiver, column = j node = sender. Each cell controls transfer j→i)

| i\j | 0 | 1 | 2 |
|---|---|---|---|
| 0 | LinearGate | CorrectGate | ThroughGate |
| 1 | CutoffGate | CutoffGate | CorrectGate |
| 2 | CorrectGate | CorrectGate | CorrectGate |

> Note: In the code, the outer loop is `i` (receiver) and the inner loop is `j` (sender) (`{i}_{j}_gate`).

### Retraining + test results (train+val → test)

#### dcl_test.py (retraining with the best trial)
- Log: `examples/CIFAR-100/evaluation/runs/dcl_3/{trial:04}/{node}_{model}/`
  - Example: `examples/CIFAR-100/evaluation/runs/dcl_3/0017/0_resnet32/`
- Checkpoints: `examples/CIFAR-100/evaluation/checkpoint/dcl_3/{trial:04}/{node}_{model}/`
  - Example: `examples/CIFAR-100/evaluation/checkpoint/dcl_3/0017/0_resnet32/`

| Node | Model | Best Top-1(%) | Best Epoch |
|---|---|---:|---:|
| 0 | resnet32  | 73.64 | 195 |
| 1 | resnet32 | 73.41 | 196 |
| 2 | wideresnet28_2 | 76.50 | 199 |

> Note: `dcl_test.py` reconstructs the graph with the best-trial configuration and evaluates on the `test` set.

#### dml_test.py (all edges ThroughGate)
- Log: `examples/CIFAR-100/evaluation/runs/dml_3/{trial:04}/{node}_{model}/`
  - Example: `examples/CIFAR-100/evaluation/runs/dml_3/0017/0_resnet32/`
- Checkpoints: `examples/CIFAR-100/evaluation/checkpoint/dml_3/{trial:04}/{node}_{model}/`
  - Example: `examples/CIFAR-100/evaluation/checkpoint/dml_3/0017/0_resnet32/`

| Node | Model | Best Top-1(%) | Best Epoch |
|---|---|---:|---:|
| 0 | resnet32  | 73.55 | 196 |
| 1 | resnet32 | 73.85 | 197 |
| 2 | wideresnet28_2 | 76.79 | 197 |

> Note: This evaluates the DML configuration where all edges are fixed to `ThroughGate`.

### Single model test results (pre_train.py, train+val → test)
- Log: `examples/CIFAR-100/evaluation/runs/pre-train/{model}/`
  - Example: `examples/CIFAR-100/evaluation/runs/pre-train/resnet32/`
- Checkpoints: `examples/CIFAR-100/evaluation/checkpoint/pre-train/{model}/`

| Model | Best Top-1(%) | Best Epoch |
|---|---:|---:|
| resnet32 | 71.44 | 197 |
| resnet110 | 73.67 | 187 |
| wideresnet28_2 | 75.56 | 198 |

### Quick Start
1) Pre-train (optional)
```bash
cd examples/CIFAR-100
uv run pre_train.py --model resnet32
uv run pre_train.py --model resnet110
uv run pre_train.py --model wideresnet28_2
```

2) Search with Optuna (DCL)
```bash
cd examples/CIFAR-100
uv run dcl_train.py --num-nodes 3 --n_trials 100 \
  --models resnet32 resnet110 wideresnet28_2 \
  --gates ThroughGate CutoffGate PositiveLinearGate NegativeLinearGate
```

3) Retraining + evaluation with the best trial (test-operation mode)
```bash
cd examples/CIFAR-100/evaluation
uv run dcl_test.py --num-nodes 3 --trial 17
# If omitted, the study's best_trial is used by default
```

### Notes
- `runs/` is the output directory for TensorBoard logs.
- `checkpoint/` stores the best models (including `pre-train`).
- For a node where all incoming edges are `CutoffGate` and `i!=0`, the corresponding model's pre-trained checkpoint is automatically loaded.


