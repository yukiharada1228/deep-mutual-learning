from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Edge:
    """
    A directed edge in the learning graph.

    ``source=None`` denotes a **label edge** (ŷ → target): the criterion is
    applied directly to the node's output and the ground-truth label.

    ``source=<int>`` denotes a **peer edge** (source → target): the criterion
    receives temperature-scaled log-softmax / softmax of the two nodes' outputs.

    Args:
        source:      Index of the source node, or ``None`` for a label edge.
        target:      Index of the target node.
        criterion:   Loss function for this edge.
        temperature: Softmax temperature (peer edges only, default: 1.0).

    Example::

        # DML: 2 mutual students
        session = Session(
            nodes,
            Edge(None, 0, nn.CrossEntropyLoss()),
            Edge(None, 1, nn.CrossEntropyLoss()),
            Edge(0, 1, nn.KLDivLoss(reduction="batchmean")),
            Edge(1, 0, nn.KLDivLoss(reduction="batchmean")),
        )

        # KD: node 0 is a frozen teacher (no incoming edges)
        session = Session(
            nodes,
            Edge(None, 1, nn.CrossEntropyLoss()),
            Edge(0, 1, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
        )

        # hybrid: node 0 teacher, nodes 1 and 2 do mutual learning
        session = Session(
            nodes,
            Edge(None, 1, nn.CrossEntropyLoss()),
            Edge(None, 2, nn.CrossEntropyLoss()),
            Edge(0, 1, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
            Edge(0, 2, nn.KLDivLoss(reduction="batchmean"), temperature=4.0),
            Edge(1, 2, nn.KLDivLoss(reduction="batchmean")),
            Edge(2, 1, nn.KLDivLoss(reduction="batchmean")),
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
