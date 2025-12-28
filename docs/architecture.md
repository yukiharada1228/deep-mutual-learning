# Architecture

This document explains the architecture and design principles of the Knowledge Transfer Graph framework.

## Overview

Knowledge Transfer Graph (KTG) is a framework for training multiple neural networks collaboratively. The framework models knowledge transfer as a directed graph where:

- **Nodes** represent neural network models
- **Edges** represent knowledge transfer paths between models
- **Gates** control the temporal dynamics of knowledge transfer

## Core Components

### 1. KnowledgeTransferGraph

The main class that orchestrates the training of multiple models. It manages:

- Training loop across all nodes
- Batch processing and forward/backward passes
- Evaluation and metric tracking
- Integration with Optuna for hyperparameter optimization

```python
class KnowledgeTransferGraph:
    def __init__(
        self,
        nodes: list[Node],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        trial=None,
    )
```

### 2. Node

A Node represents a single model in the graph. Each node contains:

- **model**: The neural network to train
- **edges**: List of edges connecting to this node (incoming edges)
- **optimizer**: Optimizer for updating model parameters
- **scheduler**: Learning rate scheduler
- **writer**: TensorBoard writer for logging
- **scaler**: Gradient scaler for mixed precision training
- **loss_meter**: Tracks average loss
- **score_meter**: Tracks average accuracy/score

```python
@dataclass
class Node:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.amp.GradScaler
    optimizer: Optimizer
    edges: list[Edge]
    loss_meter: AverageMeter
    score_meter: AverageMeter
    scheduler: LRScheduler = None
    best_score: float = 0.0
    eval: nn.Module = accuracy
    save_dir: Optional[str] = None
```

### 3. Edge

An Edge represents a knowledge transfer path. Each edge consists of:

- **criterion**: Loss function for computing the transfer loss
- **gate**: Gate module that controls the transfer weight over time

```python
class Edge(nn.Module):
    def __init__(self, criterion: nn.Module, gate: nn.Module):
        self.criterion = criterion
        self.gate = gate

    def forward(self, target_output, label, source_output, epoch, is_self_edge):
        if is_self_edge:
            loss = self.criterion(target_output, label)
            return self.gate(loss, epoch)
        else:
            # For KLDivLoss, convert logits to proper format
            if isinstance(self.criterion, nn.KLDivLoss):
                target_log_prob = F.log_softmax(target_output, dim=-1)
                source_prob = F.softmax(source_output.detach(), dim=-1)
                loss = self.criterion(target_log_prob, source_prob)
                # Sum over classes to get per-sample loss
                loss = loss.sum(dim=-1)
            else:
                loss = self.criterion(target_output, source_output)

            return self.gate(
                loss, epoch,
                student_logits=target_output,
                teacher_logits=source_output,
                label=label
            )
```

### 4. TotalLoss

Each node has a `TotalLoss` module that aggregates losses from all incoming edges. This is automatically created from the node's edges in `__post_init__`:

```python
class TotalLoss(nn.Module):
    def __init__(self, edges: list[Edge]):
        super(TotalLoss, self).__init__()
        self.incoming_edges = nn.ModuleList(edges)
    
    def forward(self, model_id, outputs, labels, epoch):
        if model_id < 0 or model_id >= len(outputs):
            raise ValueError(f"Invalid model_id: {model_id}")
        losses = []
        target_output = outputs[model_id]
        label = labels[model_id]
        for i, edge in enumerate(self.incoming_edges):
            if i == model_id:
                # Self-edge: use ground truth label
                loss = edge(target_output, label, None, epoch, True)
            else:
                # Transfer edge: use source model output
                loss = edge(target_output, None, outputs[i], epoch, False)
            losses.append(loss)
        return torch.stack(losses).sum()
```

## Training Flow

### Forward Pass

1. For each batch, all models process the input in parallel with mixed precision:
   ```python
   outputs = []
   labels = []
   for node in self.nodes:
       node.model.train()
       with torch.amp.autocast("cuda"):
           y = node.model(image)
       outputs.append(y)
       labels.append(label)
   ```

2. For each node, compute the total loss:
   ```python
   with torch.amp.autocast("cuda"):
       loss = node.total_loss(model_id, outputs, labels, epoch)
   ```

3. The total loss aggregates:
   - Self-edge loss: CrossEntropyLoss with ground truth labels (`reduction="none"`)
   - Transfer edge losses: KLDivLoss with other models' outputs (weighted by gates)
   - All losses use `reduction="none"` for per-sample computation
   - Gates apply per-sample filtering/weighting and return averaged loss

### Backward Pass

1. Scale the loss with gradient scaler (for mixed precision):
   ```python
   if loss != 0:
       node.scaler.scale(loss).backward()
       node.scaler.step(node.optimizer)
       node.optimizer.zero_grad()
       node.scaler.update()
   ```

2. Update learning rate scheduler (after epoch):
   ```python
   if node.scheduler is not None:
       node.scheduler.step()
   ```

### Evaluation

After each epoch:
1. Evaluate all models on validation set (with `model.eval()` and `torch.no_grad()`)
2. Log metrics to TensorBoard (train_loss, train_score, test_score, train_lr)
3. Save best checkpoints (if `save_dir` is specified)
4. Report to Optuna trial (if applicable, using node 0's score)

## Gates

Gates control the temporal dynamics of knowledge transfer. All gates receive per-sample losses (shape: `(batch_size,)`) and return a scalar loss.

### ThroughGate

Always transfers knowledge:
```python
def forward(self, loss, epoch, **kwargs):
    return loss.mean()  # Simple average of all samples
```

### CutoffGate

Never transfers knowledge:
```python
def forward(self, loss, epoch, **kwargs):
    return torch.zeros_like(loss[0], requires_grad=True).sum()  # Returns 0
```

### LinearGate

Gradually increases transfer from 0 to 1 over epochs:
```python
def forward(self, loss, epoch, **kwargs):
    loss_weight = epoch / (self.max_epoch - 1)
    return (loss * loss_weight).mean()
```

### CorrectGate

Filters samples based on teacher's prediction correctness (as proposed in the paper):
```python
def forward(self, loss, epoch, student_logits, teacher_logits, label, **kwargs):
    # Determine correctness of predictions
    true_s = student_logits.argmax(dim=1) == label
    true_t = teacher_logits.argmax(dim=1) == label

    # Create masks for each case
    TT = ((true_t == 1) & (true_s == 1)).float()  # Both correct
    TF = ((true_t == 1) & (true_s == 0)).float()  # Teacher correct, student wrong
    FT = ((true_t == 0) & (true_s == 1)).float()  # Teacher wrong, student correct
    FF = ((true_t == 0) & (true_s == 0)).float()  # Both wrong

    # Paper definition: TT=1, TF=1, FT=0, FF=0
    # Only transfer when teacher is correct
    mask = 1 * TT + 1 * TF + 0 * FT + 0 * FF

    return (loss * mask).mean()
```

## Loss Functions

### CrossEntropyLoss

Standard classification loss for self-edges (training with ground truth labels):
```python
criterion = nn.CrossEntropyLoss(reduction="none")  # Per-sample loss
```

### KLDivLoss

Knowledge distillation loss for transfer edges. Uses PyTorch's official implementation:
```python
criterion = nn.KLDivLoss(reduction="none")  # Per-sample loss
```

The Edge class handles the conversion of logits to the proper format:
```python
# In Edge.forward() for transfer edges:
target_log_prob = F.log_softmax(target_output, dim=-1)  # Student
source_prob = F.softmax(source_output.detach(), dim=-1)  # Teacher
loss = criterion(target_log_prob, source_prob)  # Shape: (batch, classes)
loss = loss.sum(dim=-1)  # Sum over classes → Shape: (batch,)
```

This ensures all losses have the same shape `(batch_size,)` for consistent gate processing.

## Graph Structure

In a typical setup with N nodes:

- Each node has N edges (one to each node, including itself)
- The self-edge uses `CrossEntropyLoss(reduction="none")`
- Other edges use `KLDivLoss(reduction="none")` for knowledge distillation
- Gates can be configured independently for each edge
- All gates operate on per-sample losses and return averaged loss

This creates a fully connected graph where each model can learn from all others. The use of `reduction="none"` allows gates (especially CorrectGate) to apply sample-wise filtering before averaging.

## Integration with Optuna

KTG supports hyperparameter optimization with Optuna:

1. Pass an Optuna `trial` object to `KnowledgeTransferGraph`
2. Suggest hyperparameters in the objective function (gates, models, etc.)
3. Report validation scores after each epoch
4. Use pruning to stop unpromising trials early

```python
def objective(trial):
    # Suggest hyperparameters
    gate_name = trial.suggest_categorical(
        "gate",
        ["ThroughGate", "CutoffGate", "LinearGate", "CorrectGate"]
    )
    model_name = trial.suggest_categorical("model", ["resnet32", "resnet110"])

    # Create graph
    graph = KnowledgeTransferGraph(..., trial=trial)
    best_score = graph.train()
    return best_score
```

## Design Principles

1. **Modularity**: Each component (Node, Edge, Gate) is independent and replaceable
2. **Flexibility**: Support for arbitrary graph structures and gate configurations
3. **Efficiency**: Parallel forward passes and efficient loss computation
4. **Extensibility**: Easy to add new gates, loss functions, or models

