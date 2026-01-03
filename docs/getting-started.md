# Getting Started

This guide will help you get started with Deep Collaborative Learning (DCL).

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.5.1
- CUDA-capable GPU (recommended)

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yukiharada1228/KnowledgeTransferGraph.git
cd KnowledgeTransferGraph
```

2. Install the package using `uv`:
```bash
uv sync
```

Or using `pip`:
```bash
pip install -e .
```

## Basic Usage

### Step 1: Prepare Your Data

First, prepare your training and validation data loaders:

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
```

### Step 2: Define Your Models

Create the models you want to train collaboratively:

```python
import torch.nn as nn
from dcl.models import cifar_models

num_classes = 10
model1 = cifar_models.resnet32(num_classes).cuda()
model2 = cifar_models.resnet110(num_classes).cuda()
model3 = cifar_models.wideresnet28_2(num_classes).cuda()
```

### Step 3: Create Learners

Each model becomes a learner. Define the loss functions and gates for knowledge transfer:

```python
from dcl import DistillationTrainer, Learner, build_links, gates
from dcl.models import cifar_models
from dcl.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

max_epoch = 200
num_nodes = 3

learners = []
for i in range(num_nodes):
    # Select model
    if i == 0:
        model = cifar_models.resnet32(num_classes).cuda()
    elif i == 1:
        model = cifar_models.resnet110(num_classes).cuda()
    else:
        model = cifar_models.wideresnet28_2(num_classes).cuda()

    # Define criterions with reduction="none" for per-sample loss
    # Gates will handle the averaging
    criterions = []
    for j in range(num_nodes):
        if i == j:
            criterions.append(nn.CrossEntropyLoss(reduction="none"))
        else:
            criterions.append(nn.KLDivLoss(reduction="none"))
    
    # Define gates for knowledge transfer
    gates_list = [
        gates.ThroughGate(max_epoch),   # Always transfer
        gates.CutoffGate(max_epoch),    # Never transfer
        gates.LinearGate(max_epoch),    # Gradually increase transfer
        gates.CorrectGate(max_epoch)    # Filter based on teacher correctness
    ]
    
    # Build links
    links = build_links(criterions, gates_list)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=0.0
    )
    
    # Create learner
    learner = Learner(
        model=model,
        writer=SummaryWriter(f"runs/node_{i}"),
        scaler=torch.amp.GradScaler("cuda"),
        optimizer=optimizer,
        scheduler=scheduler,
        links=links,
        loss_meter=AverageMeter(),
        score_meter=AverageMeter(),
    )
    learners.append(learner)
```

### Step 4: Create and Train

Create the DistillationTrainer and start training:

```python
trainer = DistillationTrainer(
    learners=learners,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=torch.device("cuda"),
)

best_score = trainer.train()
print(f"Best validation accuracy: {best_score:.2f}%")
```

## Understanding Gates

Gates control when and how much knowledge is transferred between models. All gates operate on per-sample losses (using `reduction="none"`) and return the averaged loss.

### Available Gate Functions

- **ThroughGate**: Always transfers knowledge (weight = 1.0)
  ```python
  loss_out = loss.mean()
  ```

- **CutoffGate**: Never transfers knowledge (weight = 0.0)
  ```python
  loss_out = 0.0
  ```

- **LinearGate**: Gradually increases transfer from 0 to 1 over epochs
  ```python
  weight = current_epoch / (max_epoch - 1)
  loss_out = (loss * weight).mean()
  ```

- **CorrectGate**: Filters samples based on teacher's prediction correctness (as described in the paper)
  ```python
  # Only transfer knowledge when teacher predicts correctly
  mask = (teacher_correct) ? 1.0 : 0.0
  loss_out = (loss * mask).mean()
  ```

## Next Steps

- Read the [Architecture](architecture.md) guide to understand the framework in detail
- Check out the [API Reference](api-reference.md) for complete API documentation
- See [Examples](examples.md) for more advanced usage patterns

