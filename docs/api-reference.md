# API Reference

Complete API documentation for the Knowledge Transfer Graph library.

## Core Classes

### DistillationTrainer

Main class for training multiple models collaboratively.

```python
class DistillationTrainer:
    def __init__(
        self,
        learners: list[Learner],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        trial: Optional[optuna.Trial] = None,
    )
```

**Parameters:**
- `learners` (list[Learner]): List of Learner instances representing models
- `max_epoch` (int): Maximum number of training epochs
- `train_dataloader` (DataLoader): DataLoader for training data
- `test_dataloader` (DataLoader): DataLoader for validation/test data
- `device` (torch.device): Device to use for training (e.g., "cuda" or "cpu")
- `trial` (Optional[optuna.Trial]): Optuna trial object for hyperparameter optimization

**Methods:**

#### `train() -> float`
Train all learners and return the best validation score.

**Returns:**
- `float`: Best validation accuracy (from node 0)

**Example:**
```python
trainer = DistillationTrainer(learners, max_epoch=200, ...)
best_score = trainer.train()
```

#### `train_on_batch(image, label, epoch, num_iter)`
Process a single training batch.

**Parameters:**
- `image` (Tensor): Input images
- `label` (Tensor): Ground truth labels
- `epoch` (int): Current epoch (0-indexed)
- `num_iter` (int): Current iteration number

#### `test_on_batch(image, label)`
Process a single validation batch.

**Parameters:**
- `image` (Tensor): Input images
- `label` (Tensor): Ground truth labels

---

### Learner

Represents a single model in the distillation process.

```python
@dataclass
class Learner:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.amp.GradScaler
    optimizer: Optimizer
    links: list[DistillationLink]
    loss_meter: AverageMeter
    score_meter: AverageMeter
    scheduler: Optional[LRScheduler] = None
    best_score: float = 0.0
    eval: nn.Module = accuracy
    save_dir: Optional[str] = None
```

**Fields:**
- `model` (nn.Module): The neural network model
- `writer` (SummaryWriter): TensorBoard writer for logging
- `scaler` (torch.amp.GradScaler): Gradient scaler for mixed precision
- `optimizer` (Optimizer): Optimizer for parameter updates
- `links` (list[DistillationLink]): List of incoming links (knowledge transfer paths)
- `loss_meter` (AverageMeter): Tracks average training loss
- `score_meter` (AverageMeter): Tracks average accuracy/score
- `scheduler` (Optional[LRScheduler]): Learning rate scheduler
- `best_score` (float): Best validation score achieved
- `eval` (nn.Module): Evaluation function (default: accuracy)
- `save_dir` (Optional[str]): Directory to save checkpoints

**Note:** The `composite_loss` field is automatically created from `links` in `__post_init__` as a `CompositeLoss` instance.

---

### DistillationLink

Represents a knowledge transfer path between models.

```python
class DistillationLink(nn.Module):
    def __init__(self, criterion: nn.Module, gate: nn.Module)
```

**Parameters:**
- `criterion` (nn.Module): Loss function (e.g., CrossEntropyLoss, KLDivLoss)
- `gate` (nn.Module): Gate module that controls transfer weight

**Methods:**

#### `forward(target_output, label, source_output, epoch, is_self_link) -> Tensor`
Compute the weighted loss for this link.

**Parameters:**
- `target_output` (Tensor): Output from the target model
- `label` (Tensor): Ground truth labels (for self-links)
- `source_output` (Tensor): Output from the source model (for transfer links)
- `epoch` (int): Current epoch (0-indexed)
- `is_self_link` (bool): Whether this is a self-link

**Returns:**
- `Tensor`: Weighted loss value

---

### build_links

Helper function to create a list of DistillationLink instances.

```python
def build_links(
    criterions: list[nn.Module],
    gates: list[nn.Module]
) -> list[DistillationLink]
```

**Parameters:**
- `criterions` (list[nn.Module]): List of loss functions
- `gates` (list[nn.Module]): List of gate modules

**Returns:**
- `list[DistillationLink]`: List of DistillationLink instances

**Broadcasting:**
- If `criterions` has length 1 and `gates` has length N>1, `criterions` is broadcast to length N
- If `gates` has length 1 and `criterions` has length N>1, `gates` is broadcast to length N
- Otherwise, lengths must match

**Example:**
```python
criterions = [
    nn.CrossEntropyLoss(reduction="none"),
    nn.KLDivLoss(reduction="none"),
    nn.KLDivLoss(reduction="none")
]
gates = [ThroughGate(200), ThroughGate(200), CutoffGate(200)]
links = build_links(criterions, gates)
```

---

## Gates

All gates inherit from `nn.Module` and implement the same interface. Gates receive per-sample losses (shape: `(batch_size,)`) and return a scalar loss.

### ThroughGate

Always transfers knowledge (weight = 1.0).

```python
class ThroughGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int, **kwargs) -> Tensor
```

**Implementation:** Returns `loss.mean()` - simple average of all samples.

### CutoffGate

Never transfers knowledge (weight = 0.0).

```python
class CutoffGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int, **kwargs) -> Tensor
```

**Implementation:** Returns `torch.zeros_like(loss[0], requires_grad=True).sum()` - effectively zero.

### LinearGate

Gradually increases transfer from 0 to 1 over epochs.

```python
class LinearGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int, **kwargs) -> Tensor
```

**Weight formula:** `weight = epoch / (max_epoch - 1)`

**Implementation:** Returns `(loss * weight).mean()`

### CorrectGate

Filters samples based on teacher's prediction correctness.

```python
class CorrectGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(
        self,
        loss: Tensor,
        epoch: int,
        teacher_logits: Tensor,
        label: Tensor,
        **kwargs
    ) -> Tensor
```

**Parameters:**
- `loss` (Tensor): Per-sample losses, shape `(batch_size,)`
- `epoch` (int): Current epoch (0-indexed)
- `teacher_logits` (Tensor): Teacher model's output logits
- `label` (Tensor): Ground truth labels

**Filtering logic:**
- Teacher correct: weight = 1.0 (approximated by mask)
- Teacher wrong: weight = 0.0

Only transfers knowledge when the teacher makes correct predictions.

**Implementation:** Returns `(loss * mask).mean()` where mask is based on teacher correctness.

---

## Loss Functions

KTG uses PyTorch's official loss functions with `reduction="none"` for per-sample loss computation.

### CrossEntropyLoss

Standard classification loss for self-links (training with ground truth labels):

```python
criterion = nn.CrossEntropyLoss(reduction="none")
```

**Output shape:** `(batch_size,)` - per-sample loss

### KLDivLoss

Knowledge distillation loss for transfer links. Uses PyTorch's official implementation:

```python
criterion = nn.KLDivLoss(reduction="none")
```

**Important:** The DistillationLink class automatically converts logits to the proper format:
```python
# Student output → log-probabilities
target_log_prob = F.log_softmax(target_output, dim=-1)

# Teacher output → probabilities
source_prob = F.softmax(source_output.detach(), dim=-1)

# Compute loss
loss = criterion(target_log_prob, source_prob)  # Shape: (batch, classes)

# Sum over classes to get per-sample loss
loss = loss.sum(dim=-1)  # Shape: (batch,)
```

**Output shape:** `(batch_size,)` - per-sample loss (after summing over classes)

This ensures all losses have consistent shape for gate processing.

---

## Utility Functions

### AverageMeter

Tracks running average of values.

```python
class AverageMeter:
    def __init__(self)
    def update(self, val: float, n: int = 1)
    def reset(self)
    @property
    def avg(self) -> float
```

**Example:**
```python
from dcl.utils import AverageMeter

meter = AverageMeter()
meter.update(0.95, n=100)
meter.update(0.90, n=50)
print(meter.avg)  # Average of all values
```

### save_checkpoint

Save model checkpoint.

```python
def save_checkpoint(
    model: nn.Module,
    save_dir: str,
    epoch: int,
    is_best: bool = False
)
```

**Parameters:**
- `model` (nn.Module): Model to save
- `save_dir` (str): Directory to save checkpoint
- `epoch` (int): Current epoch number
- `is_best` (bool): If True, save as `best_checkpoint.pkl`

### load_checkpoint

Load model checkpoint.

```python
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str
)
```

**Parameters:**
- `model` (nn.Module): Model to load weights into
- `checkpoint_path` (str): Path to the checkpoint file

### set_seed

Set random seed for reproducibility.

```python
def set_seed(seed: int)
```

**Parameters:**
- `seed` (int): Random seed value

### accuracy

Compute top-k accuracy.

```python
def accuracy(
    output: Tensor,
    target: Tensor,
    topk: tuple = (1,)
) -> list[float]
```

**Parameters:**
- `output` (Tensor): Model predictions (logits)
- `target` (Tensor): Ground truth labels
- `topk` (tuple): Tuple of k values (e.g., (1, 5))

**Returns:**
- `list[float]`: List of top-k accuracies

### WorkerInitializer

Initialize DataLoader workers with a fixed seed.

```python
class WorkerInitializer:
    def __init__(self, seed: int)
    def worker_init_fn(self, worker_id: int)
```

**Example:**
```python
from dcl.utils import WorkerInitializer

loader = DataLoader(
    dataset,
    worker_init_fn=WorkerInitializer(42).worker_init_fn
)
```

---

## Models

### CIFAR Models

Pre-defined models for CIFAR-10/100 datasets.

```python
from dcl.models import cifar_models

# ResNet variants
model = cifar_models.resnet18(num_classes=10)
model = cifar_models.resnet20(num_classes=10)
model = cifar_models.resnet32(num_classes=10)
model = cifar_models.resnet34(num_classes=10)
model = cifar_models.resnet44(num_classes=10)
model = cifar_models.resnet50(num_classes=10)
model = cifar_models.resnet56(num_classes=10)
model = cifar_models.resnet110(num_classes=10)
model = cifar_models.resnet1202(num_classes=10)

# Wide ResNet
model = cifar_models.wideresnet28_2(num_classes=10)
```

**Parameters:**
- `num_classes` (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)

---

## Datasets

### CIFAR-10 Dataset

```python
from dcl.dataset.cifar_datasets.cifar10 import get_datasets

train_dataset, val_dataset = get_datasets(use_test_mode=False)
```

**Parameters:**
- `use_test_mode` (bool): If `True`, returns (train+val, test) datasets. If `False`, returns (train, val) datasets (default: `False`)

**Returns:**
- `train_dataset`: Training dataset (or train+val if `use_test_mode=True`)
- `val_dataset`: Validation dataset (or test dataset if `use_test_mode=True`)

**Note:** The datasets automatically download CIFAR-10 data to `data/` directory and apply appropriate transforms (normalization, augmentation).

### CIFAR-100 Dataset

```python
from dcl.dataset.cifar_datasets.cifar100 import get_datasets

train_dataset, val_dataset = get_datasets(use_test_mode=False)
```

**Parameters:**
- `use_test_mode` (bool): If `True`, returns (train+val, test) datasets. If `False`, returns (train, val) datasets (default: `False`)

**Returns:**
- `train_dataset`: Training dataset (or train+val if `use_test_mode=True`)
- `val_dataset`: Validation dataset (or test dataset if `use_test_mode=True`)

**Note:** The datasets automatically download CIFAR-100 data to `data/` directory and apply appropriate transforms (normalization, augmentation).

