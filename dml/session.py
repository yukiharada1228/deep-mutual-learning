from __future__ import annotations

import torch

from .graph import Edge, Node


class Session:
    def __init__(self, nodes: list[Node], *edges: Edge):
        self.nodes = nodes
        self.edges = edges
        self._targets = {e.target for e in edges}

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
