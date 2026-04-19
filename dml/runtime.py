from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(eq=False)
class DMLNode:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

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


class DMLSession:
    def __init__(self, nodes: list[DMLNode], losses: list[nn.Module]):
        if len(nodes) != len(losses):
            raise ValueError(
                f"len(nodes)={len(nodes)} must equal len(losses)={len(losses)}"
            )
        self.nodes = nodes
        self.losses = losses

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def forward_all(self, image: torch.Tensor, device_type: str) -> list[torch.Tensor]:
        return [node.forward(image, device_type) for node in self.nodes]

    def compute_losses(
        self,
        outputs: list[torch.Tensor],
        label: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [loss(outputs, label) for loss in self.losses]

    def optimize_all(self, losses: list[torch.Tensor]):
        for node, loss in zip(self.nodes, losses):
            node.optimize(loss)

    def step_schedulers(self):
        for node in self.nodes:
            node.step_scheduler()
