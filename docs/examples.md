# Examples

This document provides practical examples of using Knowledge Transfer Graph.

## Example 1: Basic 3-Node Graph

Train three models collaboratively with simple gate configurations.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcl import DistillationTrainer, Learner, build_links, gates
from dcl.models import cifar_models
from dcl.dataset.cifar_datasets.cifar10 import get_datasets
from dcl.utils import AverageMeter, WorkerInitializer, set_seed

# Set seed for reproducibility
set_seed(42)

# Prepare data
train_dataset, val_dataset = get_datasets(use_test_mode=False)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=WorkerInitializer(42).worker_init_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=WorkerInitializer(42).worker_init_fn,
)

# Create learners
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
    
    # Define criterions with reduction="none" for per-sample loss
    criterions = []
    for j in range(num_nodes):
        if i == j:
            criterions.append(nn.CrossEntropyLoss(reduction="none"))
        else:
            criterions.append(nn.KLDivLoss(reduction="none"))
    
    # Define gates: ThroughGate for all links
    gates_list = [gates.ThroughGate(max_epoch) for _ in range(num_nodes)]
    
    # Build links
    links = build_links(criterions, gates_list)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=0.0
    )
    
    # Create learner
    learner = Learner(
        model=model,
        writer=SummaryWriter(f"runs/basic_example/node_{i}"),
        scaler=torch.amp.GradScaler("cuda"),
        optimizer=optimizer,
        scheduler=scheduler,
        links=links,
        loss_meter=AverageMeter(),
        score_meter=AverageMeter(),
    )
    learners.append(learner)

# Create and train
trainer = DistillationTrainer(
    learners=learners,
    max_epoch=max_epoch,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
)

best_score = trainer.train()
print(f"Best validation accuracy: {best_score:.2f}%")
```

## Example 2: Temporal Gate Scheduling

Use different gates to control knowledge transfer over time.

```python
# ... (same setup as Example 1) ...

# Define gates with temporal scheduling
gates_list = [
    gates.ThroughGate(max_epoch),    # Self-link: always on
    gates.LinearGate(max_epoch),     # Gradually increase transfer
    gates.CorrectGate(max_epoch),    # Filter by teacher correctness
]

links = build_links(criterions, gates_list)
```

This configuration:
- Always uses ground truth labels for training (self-edge with ThroughGate)
- Gradually increases knowledge transfer from model 1 (LinearGate)
- Filters knowledge transfer from model 2 based on teacher correctness (CorrectGate)

## Example 3: Optuna Hyperparameter Optimization

Use Optuna to find optimal gate and model configurations.

```python
import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.study import MaxTrialsCallback

def objective(trial):
    set_seed(42)
    
    # Prepare data
    train_dataset, val_dataset = get_datasets(use_test_mode=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=WorkerInitializer(42).worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=WorkerInitializer(42).worker_init_fn,
    )
    
    max_epoch = 200
    num_nodes = 3
    num_classes = 10
    
    learners = []
    for i in range(num_nodes):
        # Suggest gate for each link
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss(reduction="none"))
            else:
                criterions.append(nn.KLDivLoss(reduction="none"))

            # Suggest gate type
            gate_name = trial.suggest_categorical(
                f"{i}_{j}_gate",
                ["ThroughGate", "CutoffGate", "LinearGate", "CorrectGate"]
            )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)
        
        # Suggest model for non-primary learners
        if i == 0:
            model_name = "resnet32"
        else:
            model_name = trial.suggest_categorical(
                f"{i}_model",
                ["resnet32", "resnet110", "wideresnet28_2"]
            )
        
        model = getattr(cifar_models, model_name)(num_classes).cuda()
        
        links = build_links(criterions, gates_list)
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch, eta_min=0.0
        )
        
        learner = Learner(
            model=model,
            writer=SummaryWriter(f"runs/optuna/{trial.number:04}/node_{i}"),
            scaler=torch.amp.GradScaler("cuda"),
            optimizer=optimizer,
            scheduler=scheduler,
            links=links,
            loss_meter=AverageMeter(),
            score_meter=AverageMeter(),
        )
        learners.append(learner)
    
    trainer = DistillationTrainer(
        learners=learners,
        max_epoch=max_epoch,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        trial=trial,  # Pass trial for pruning
    )
    
    best_score = trainer.train()
    return best_score

# Create Optuna study
study_name = "ktg_optimization"
storage = JournalStorage(JournalFileStorage(f"optuna/{study_name}/optuna.log"))
sampler = optuna.samplers.TPESampler(multivariate=True)
pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(
    storage=storage,
    study_name=study_name,
    sampler=sampler,
    pruner=pruner,
    direction="maximize",
    load_if_exists=True,
)

# Run optimization
study.optimize(
    objective,
    n_trials=100,
    callbacks=[MaxTrialsCallback(100, states=None)],
)
```

## Example 4: Custom Model Integration

Use your own models with KTG.

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Use custom model
model1 = MyCustomModel(num_classes=10).cuda()
model2 = MyCustomModel(num_classes=10).cuda()

# ... rest of the setup is the same ...
```

## Example 5: Pre-trained Model Initialization

Initialize some models with pre-trained weights. This is useful when you want to start training from a checkpoint:

```python
from dcl.utils import load_checkpoint

# ... create learners ...

for i, learner in enumerate(learners):
    if i > 0:  # Initialize non-primary learners with pre-trained weights
        # Get model name (e.g., "ResNet32", "ResNet110")
        model_name = learner.model.__class__.__name__
        load_checkpoint(
            model=learner.model,
            save_dir=f"checkpoint/pre-train/{model_name}",
            is_best=True,
        )
```

**Note:** The checkpoint file should be saved as `best_checkpoint.pkl` in the specified directory when using `is_best=True`.

## Example 6: Custom Loss Function

Create a custom loss function for knowledge transfer.

```python
import torch.nn as nn
import torch.nn.functional as F

class CustomDistillationLoss(nn.Module):
    def __init__(self, T=4.0):
        super().__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, student_output, teacher_output):
        # Soft targets with temperature scaling
        student_log_prob = F.log_softmax(student_output / self.T, dim=1)
        teacher_prob = F.softmax(teacher_output.detach() / self.T, dim=1)

        # Compute KL divergence per sample
        loss = self.kl_div(student_log_prob, teacher_prob)

        # Sum over classes and scale by T^2
        loss = loss.sum(dim=1) * (self.T ** 2)

        return loss  # Shape: (batch_size,)

# Use custom loss
criterions = []
for j in range(num_nodes):
    if i == j:
        criterions.append(nn.CrossEntropyLoss(reduction="none"))
    else:
        criterions.append(CustomDistillationLoss(T=4.0))
```

## Example 7: Asymmetric Graph

Create an asymmetric graph where not all models transfer to each other. Note that in the current implementation, all learners must have the same number of links (one for each learner including itself). To create asymmetric transfer, use `CutoffGate` to disable specific transfers:

```python
# Model 0 receives from all models
# Model 1 receives from all models (but can use CutoffGate to disable specific transfers)
# Model 2 receives from models 0 and 1 (uses CutoffGate for self-to-self transfer)

learners = []
for i in range(3):
    criterions = []
    gates_list = []
    
    for j in range(3):
        if i == j:
            criterions.append(nn.CrossEntropyLoss(reduction="none"))
        else:
            criterions.append(nn.KLDivLoss(reduction="none"))
        
        if i == 0:
            # Model 0: receives from all
            gates_list.append(gates.ThroughGate(max_epoch))
        elif i == 1:
            # Model 1: receives from all
            gates_list.append(gates.ThroughGate(max_epoch))
        else:  # i == 2
            # Model 2: receives from models 0 and 1, but not from itself (use CutoffGate)
            if j == 2:
                gates_list.append(gates.CutoffGate(max_epoch))
            else:
                gates_list.append(gates.ThroughGate(max_epoch))
    
    links = build_links(criterions, gates_list)
    # ... create learner ...
```

## Example 8: Monitoring and Logging

Access training metrics during training.

```python
# After training, access metrics from learners
for i, learner in enumerate(learners):
    print(f"Learner {i}:")
    print(f"  Best score: {learner.best_score:.2f}%")
    print(f"  Final loss: {learner.loss_meter.avg:.4f}")
    print(f"  Final score: {learner.score_meter.avg:.2f}%")

# View TensorBoard logs
# tensorboard --logdir runs/
```

## Tips and Best Practices

1. **Seed Setting**: Always set seeds for reproducibility:
   ```python
   set_seed(42)
   ```

2. **Mixed Precision**: The framework uses automatic mixed precision. Ensure your models support it.

3. **Memory Management**: For large models, consider reducing batch size or using gradient accumulation.

4. **Gate Selection**:
   - Use `ThroughGate` for stable knowledge transfer
   - Use `LinearGate` when you want gradual knowledge transfer that increases over time
   - Use `CorrectGate` to filter samples based on teacher correctness (as proposed in the paper)
   - Use `CutoffGate` to disable specific transfer paths

5. **Loss Functions**: Always use `reduction="none"` for both CrossEntropyLoss and KLDivLoss:
   ```python
   nn.CrossEntropyLoss(reduction="none")
   nn.KLDivLoss(reduction="none")
   ```
   Gates will handle the averaging after applying their respective weighting/filtering.

6. **Optuna Pruning**: Use pruning to stop unpromising trials early and save computation.

