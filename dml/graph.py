from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Edge:
    """
    A directed edge in the learning graph.

    ``source=None`` denotes a **supervision edge** (ŷ → target): the criterion is
    applied directly to the node's output and the ground-truth label.

    ``source=<int>`` denotes a **distillation edge** (source → target): the criterion
    receives temperature-scaled log-softmax / softmax of the two nodes' outputs.

    Args:
        source:      Index of the source node, or ``None`` for a supervision edge.
        target:      Index of the target node.
        criterion:   Loss function for this edge.
        temperature: Softmax temperature (distillation edges only, default: 1.0).

    Example::

        # DML: 2 mutual students
        graph = Graph(
            nodes,
            [
                Edge(None, 0, nn.CrossEntropyLoss()),
                Edge(None, 1, nn.CrossEntropyLoss()),
                Edge(0, 1, nn.KLDivLoss(reduction="batchmean")),
                Edge(1, 0, nn.KLDivLoss(reduction="batchmean")),
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

        # hybrid: node 0 teacher, nodes 1 and 2 do mutual learning
        graph = Graph(
            nodes,
            [
                Edge(None, 1, nn.CrossEntropyLoss()),
                Edge(None, 2, nn.CrossEntropyLoss()),
                Edge(0, 1, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
                Edge(0, 2, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
                Edge(1, 2, nn.KLDivLoss(reduction="batchmean")),
                Edge(2, 1, nn.KLDivLoss(reduction="batchmean")),
            ],
        )
    """

    def __init__(
        self,
        source: int | None,
        target: int,
        criterion: nn.Module,
        temperature: float = 1.0,
    ):
        self.source = source
        self.target = target
        self.criterion = criterion
        self.temperature = temperature

    def compute(self, outputs: list[torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        if self.source is None:
            return self.criterion(outputs[self.target], label)

        T = self.temperature
        return self.criterion(
            F.log_softmax(outputs[self.target] / T, dim=-1),
            F.softmax(outputs[self.source].detach() / T, dim=-1),
        ) * (T**2)


@dataclass(eq=False)
class Node:
    model: nn.Module
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scaler: torch.amp.GradScaler | None = None

    @property
    def lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def forward(self, image: torch.Tensor, device_type: str) -> torch.Tensor:
        with torch.amp.autocast(device_type=device_type):
            return self.model(image)

    def optimize(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def step_scheduler(self):
        self.scheduler.step()


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes = nodes
        self.edges = edges
        self._targets = {e.target for e in edges}
        self._distill_counts = Counter(e.target for e in edges if e.source is not None)

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

    def forward_all(self, image: torch.Tensor, device_type: str) -> list[torch.Tensor]:
        outputs = []
        for i, node in enumerate(self.nodes):
            if self.is_teacher(i):
                with torch.no_grad():
                    outputs.append(node.forward(image, device_type))
            else:
                outputs.append(node.forward(image, device_type))
        return outputs

    def compute_losses(
        self,
        outputs: list[torch.Tensor],
        label: torch.Tensor,
    ) -> list[torch.Tensor | None]:
        losses: list[torch.Tensor | None] = [None] * len(self.nodes)
        for edge in self.edges:
            edge_loss = edge.compute(outputs, label)
            if edge.source is not None:
                edge_loss = edge_loss / self._distill_counts[edge.target]
            i = edge.target
            losses[i] = edge_loss if losses[i] is None else losses[i] + edge_loss
        return losses

    def optimize_all(self, losses: list[torch.Tensor | None]):
        for node, loss in zip(self.nodes, losses):
            if loss is not None:
                node.optimize(loss)

    def step_schedulers(self):
        for i, node in enumerate(self.nodes):
            if not self.is_teacher(i) and node.scheduler is not None:
                node.step_scheduler()
