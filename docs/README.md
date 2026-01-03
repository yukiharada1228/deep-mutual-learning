# Deep Collaborative Learning Documentation

Welcome to the Deep Collaborative Learning (DCL) documentation. This library implements the "Knowledge Transfer Graph for Deep Collaborative Learning" framework, as described in the [ACCV 2020 paper](https://openaccess.thecvf.com/content/ACCV2020/html/Minami_Knowledge_Transfer_Graph_for_Deep_Collaborative_Learning_ACCV_2020_paper.html).

## Overview

Deep Collaborative Learning (DCL) framework for training multiple neural networks collaboratively, where each network can learn from others through knowledge transfer. The framework uses a structure where learners represent models and links control the knowledge transfer between them.

### Key Features

- **Graph-based Architecture**: Flexible graph structure for modeling knowledge transfer between multiple models
- **Temporal Control**: Gates that control knowledge transfer over training epochs
- **Multiple Loss Functions**: Support for various loss functions including KL divergence for knowledge distillation
- **Optuna Integration**: Built-in support for hyperparameter optimization
- **TensorBoard Logging**: Automatic logging of training metrics

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation and quick start guide
- **[Architecture](architecture.md)**: Detailed explanation of the framework architecture
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Examples](examples.md)**: Usage examples and tutorials

## Quick Example

```python
from dcl import DistillationTrainer, Learner, build_links, gates
from dcl.models import cifar_models
from dcl.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Create learners (models)
max_epoch = 200
num_nodes = 3
num_classes = 10

learners = []
for i in range(num_nodes):
    # Select model
    if i == 0:
        model = cifar_models.resnet32(num_classes).cuda()
    elif i == 1:
        model = cifar_models.resnet110(num_classes).cuda()
    else:
        model = cifar_models.wideresnet28_2(num_classes).cuda()

    # Define criterions (using reduction="none" for gate functions)
    criterions = []
    for j in range(num_nodes):
        if i == j:
            criterions.append(nn.CrossEntropyLoss(reduction="none"))
        else:
            criterions.append(nn.KLDivLoss(reduction="none"))
    
    # Define gates
    gates_list = [gates.ThroughGate(max_epoch) for _ in range(num_nodes)]
    
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
    
    # Create node
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

# Create and train the graph
# Create and train
trainer = DistillationTrainer(
    learners=learners,
    max_epoch=max_epoch,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    device=torch.device("cuda"),
)
best_score = trainer.train()
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{minami2020knowledge,
  title={Knowledge Transfer Graph for Deep Collaborative Learning},
  author={Minami, Soma and Hirakawa, Tsubasa and Yamashita, Takayoshi and Fujiyoshi, Hironobu},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2020}
}
```

## License

See the [LICENSE](../LICENSE) file for details.

