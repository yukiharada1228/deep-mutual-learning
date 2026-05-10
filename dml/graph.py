from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.checkpoint import load_checkpoint


def _detach(x: Any) -> Any:
    """Recursively detach tensors (handles lists/tuples of tensors too)."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return type(x)(_detach(v) for v in x)
    return x


def default_model_inputs(batch: dict) -> tuple[Any, ...]:
    """Return the default model inputs for image classification batches."""
    return (batch["image"],)


class Edge:
    """
    A directed edge in the learning graph.

    ``source=None`` denotes a **supervision edge** (ŷ → target): the criterion is
    applied directly to the node's output and the ground-truth label from the batch.

    ``source=<int>`` denotes a **distillation edge** (source → target).
    When ``temperature`` is set, the edge applies temperature-scaled
    log-softmax / softmax preprocessing before passing to the criterion
    (standard KD).  When ``temperature`` is ``None``, the raw outputs are
    forwarded to the criterion as-is (for contrastive / arbitrary distillation).

    Args:
        source:      Index of the source node, or ``None`` for a supervision edge.
        target:      Index of the target node.
        criterion:   Loss function for this edge.
        temperature: Softmax temperature for distillation edges.
                     ``None`` (default) disables preprocessing.
        weight:      Scalar multiplier for the loss (default: 1.0).

    Example::

        # DML: 2 mutual students
        graph = Graph(
            nodes,
            [
                Edge(None, 0, nn.CrossEntropyLoss()),
                Edge(None, 1, nn.CrossEntropyLoss()),
                Edge(0, 1, nn.KLDivLoss(reduction="batchmean"), temperature=1.0),
                Edge(1, 0, nn.KLDivLoss(reduction="batchmean"), temperature=1.0),
            ],
        )

        # KD: node 0 is a frozen teacher (no incoming edges)
        graph = Graph(
            nodes,
            [
                Edge(None, 1, nn.CrossEntropyLoss()),
                Edge(0, 1, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
            ],
        )

        # SimCLR: self-supervised contrastive learning
        graph = Graph(
            [
                Node(
                    model=model,
                    optimizer=optimizer,
                    scheduler_interval="step",
                    model_input_fn=lambda batch: (
                        batch["views"][0],
                        batch["views"][1],
                    ),
                )
            ],
            [Edge(None, 0, NTXentLoss(batch_size=512, temperature=0.5))],
        )

        # DisCO: contrastive distillation (no temperature preprocessing)
        graph = Graph(
            [
                Node(
                    model=teacher,
                    model_input_fn=lambda batch: (
                        batch["views"][0],
                        batch["views"][1],
                    ),
                ),
                Node(
                    model=student,
                    optimizer=optimizer,
                    scheduler_interval="step",
                    model_input_fn=lambda batch: (
                        batch["views"][0],
                        batch["views"][1],
                    ),
                ),
            ],
            [
                Edge(None, 1, NTXentLoss(batch_size=512, temperature=0.5)),
                Edge(0, 1, DisCOLoss(), weight=0.5),
            ],
        )
    """

    def __init__(
        self,
        source: int | None,
        target: int,
        criterion: nn.Module,
        temperature: float | None = None,
        weight: float = 1.0,
    ):
        self.source = source
        self.target = target
        self.criterion = criterion
        self.temperature = temperature
        self.weight = weight

    def compute(self, outputs: list, batch: dict) -> torch.Tensor:
        if self.source is None:
            # Supervision edge
            loss = self.criterion(outputs[self.target], batch.get("label"))
        elif self.temperature is not None:
            # Classification-style distillation with temperature scaling
            T = self.temperature
            loss = self.criterion(
                F.log_softmax(outputs[self.target] / T, dim=-1),
                F.softmax(_detach(outputs[self.source]) / T, dim=-1),
            ) * (T**2)
        else:
            # Generic distillation — criterion handles everything
            loss = self.criterion(outputs[self.target], _detach(outputs[self.source]))
        return self.weight * loss


@dataclass(eq=False)
class Node:
    model: nn.Module
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scaler: torch.amp.GradScaler | None = None
    checkpoint_path: str | None = None
    eval_fn: Any | None = None  # (model, device) → float
    scheduler_interval: str = "epoch"  # "epoch" or "step"
    model_input_fn: Callable[[dict], tuple[Any, ...]] = field(
        default=default_model_inputs
    )

    def __post_init__(self):
        if self.scheduler_interval not in ("epoch", "step"):
            raise ValueError(
                "scheduler_interval must be either 'epoch' or 'step', "
                f"got {self.scheduler_interval!r}"
            )
        if not callable(self.model_input_fn):
            raise TypeError("model_input_fn must be callable")

    @property
    def lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def forward(self, batch: dict, device_type: str) -> Any:
        model_args = self.model_input_fn(batch)
        with torch.amp.autocast(device_type=device_type):
            return self.model(*model_args)

    def optimize(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def step_scheduler(self):
        if self.scheduler:
            self.scheduler.step()


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes = nodes
        self.edges = edges
        self._targets = {e.target for e in edges}
        self._distill_counts = Counter(e.target for e in edges if e.source is not None)
        for i, node in enumerate(self.nodes):
            if node.checkpoint_path is not None and self.is_teacher(i):
                load_checkpoint(node.model, node.checkpoint_path)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def is_teacher(self, i: int) -> bool:
        return i not in self._targets

    def train_all(self):
        for i, node in enumerate(self.nodes):
            node.model.eval() if self.is_teacher(i) else node.model.train()

    def eval_all(self):
        for node in self.nodes:
            node.model.eval()

    def forward_all(self, batch: dict, device_type: str) -> list:
        outputs = []
        for i, node in enumerate(self.nodes):
            if self.is_teacher(i):
                with torch.no_grad():
                    outputs.append(node.forward(batch, device_type))
            else:
                outputs.append(node.forward(batch, device_type))
        return outputs

    def compute_losses(
        self,
        outputs: list,
        batch: dict,
    ) -> list[torch.Tensor | None]:
        losses: list[torch.Tensor | None] = [None] * len(self.nodes)
        for edge in self.edges:
            edge_loss = edge.compute(outputs, batch)
            if edge.source is not None:
                edge_loss = edge_loss / self._distill_counts[edge.target]
            i = edge.target
            losses[i] = edge_loss if losses[i] is None else losses[i] + edge_loss
        return losses

    def optimize_all(self, losses: list[torch.Tensor | None]):
        for node, loss in zip(self.nodes, losses):
            if loss is not None:
                node.optimize(loss)

    def step_schedulers(self, interval: str):
        for i, node in enumerate(self.nodes):
            if (
                not self.is_teacher(i)
                and node.scheduler is not None
                and node.scheduler_interval == interval
            ):
                node.step_scheduler()
