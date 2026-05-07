import logging
import time

import torch
import torchvision
from cosine_warmup import get_cosine_schedule_with_warmup
from knn_eval import evaluate_knn
from lars import LARS
from models import cifar_models
from models.simclr_model import SimCLR
from torch.utils.data import DataLoader
from torchvision import transforms
from transform import SimCLRTransforms

from dml import (Callback, CheckpointCallback, EpochState, TensorBoardCallback,
                 Trainer)
from dml.graph import Edge, Graph, Node
from dml.utils import AverageMeter, WorkerInitializer

CIFAR10_NUM_CLASSES = 10
DEFAULT_NUM_WORKERS = 10

logger = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataloaders ───────────────────────────────────────────────────────────────


def create_simclr_train_dataloader(
    batch_size: int,
    seed: int,
    color_jitter_strength: float = 0.5,
    include_blur: bool = False,
    data_root: str = "data",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=SimCLRTransforms(
            input_size=32, s=color_jitter_strength, include_blur=include_blur
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=WorkerInitializer(seed).worker_init_fn,
    )


def create_knn_dataloaders(
    data_root: str = "data",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    knn_kwargs = dict(
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return (
        DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=transform
            ),
            **knn_kwargs,
        ),
        DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform
            ),
            **knn_kwargs,
        ),
    )


# ── Model / Optimizer / Scheduler / Scaler ────────────────────────────────────


def create_simclr_model(
    model_name: str,
    device: torch.device,
    projection_dim: int = 128,
    num_proj_layers: int = 2,
) -> SimCLR:
    return SimCLR(
        lambda: getattr(cifar_models, model_name)(CIFAR10_NUM_CLASSES),
        out_dim=projection_dim,
        num_proj_layers=num_proj_layers,
    ).to(device)


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    wd: float,
    momentum: float = 0.9,
) -> LARS:
    return LARS(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        momentum=momentum,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def create_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))


# ── SimCLR Framework Extensions ───────────────────────────────────────────────


class SimCLRNode(Node):
    """Node that handles SimCLR's two-view forward pass."""

    def forward(
        self, views: list[torch.Tensor], device_type: str
    ) -> list[torch.Tensor]:
        with torch.amp.autocast(device_type=device_type):
            return self.model(views[0], views[1])


class SimCLREdge(Edge):
    """Edge for NT-Xent loss; unpacks [z1, z2] output from SimCLRNode."""

    def compute(self, outputs: list, label) -> torch.Tensor:
        z1, z2 = outputs[self.target]
        return self.criterion(z1, z2)


class DisCOEdge(Edge):
    """Distillation edge for DisCO loss with an optional loss weight."""

    def __init__(self, source: int, target: int, criterion, weight: float = 1.0):
        super().__init__(source, target, criterion)
        self.weight = weight

    def compute(self, outputs: list, label) -> torch.Tensor:
        s_z1, s_z2 = outputs[self.target]
        t_z1, t_z2 = outputs[self.source]
        return self.weight * self.criterion(s_z1, s_z2, t_z1, t_z2)


class SimCLRGraph(Graph):
    """Graph that passes a two-element view list through all nodes."""

    def forward_all(self, views: list[torch.Tensor], device_type: str) -> list:
        outputs = []
        for i, node in enumerate(self.nodes):
            if self.is_teacher(i):
                with torch.no_grad():
                    outputs.append(node.forward(views, device_type))
            else:
                outputs.append(node.forward(views, device_type))
        return outputs


class SimCLRTensorBoardCallback(TensorBoardCallback):
    """TensorBoardCallback that logs train_lr, train_loss, and knn_top1."""

    def on_epoch_end(self, session, state: EpochState) -> None:
        for model_id, writer in enumerate(self.writers):
            if writer is None:
                continue
            writer.add_scalar("train_lr", state.learning_rates[model_id], state.epoch)
            writer.add_scalar("train_loss", state.train_losses[model_id], state.epoch)
            if state.val_scores[model_id] > 0:
                writer.add_scalar("knn_top1", state.val_scores[model_id], state.epoch)


class SimCLRTrainer(Trainer):
    """Trainer for SimCLR with per-batch scheduling and KNN evaluation."""

    def __init__(
        self,
        graph: SimCLRGraph,
        device: torch.device,
        knn_train_loader: DataLoader,
        knn_test_loader: DataLoader,
        knn_k: int = 20,
        knn_temperature: float = 0.07,
        num_classes: int = CIFAR10_NUM_CLASSES,
        knn_eval_freq: int = 1,
        callbacks: list[Callback] | None = None,
    ):
        super().__init__(graph, device, score_fn=lambda o, t: 0.0, callbacks=callbacks)
        self.knn_train_loader = knn_train_loader
        self.knn_test_loader = knn_test_loader
        self.knn_k = knn_k
        self.knn_temperature = knn_temperature
        self.num_classes = num_classes
        self.knn_eval_freq = knn_eval_freq

    def fit(
        self, train_dataloader: DataLoader, val_dataloader=None, epochs: int = 1000
    ) -> None:
        device_type = self.device.type

        for callback in self.callbacks:
            callback.on_train_begin(self.graph)

        logger.info("Training started")
        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d", epoch, epochs)
            start_time = time.time()

            train_loss_meters = [AverageMeter() for _ in self.graph]

            self.graph.train_all()
            for images, _ in train_dataloader:
                views = [images[0].to(self.device), images[1].to(self.device)]
                outputs = self.graph.forward_all(views, device_type)
                losses = self.graph.compute_losses(outputs, None)
                self.graph.optimize_all(losses)
                self.graph.step_schedulers()  # per-batch for cosine-warmup schedule

                for model_id, loss in enumerate(losses):
                    if loss is not None:
                        train_loss_meters[model_id].update(
                            loss.item(), views[0].size(0)
                        )

            train_losses = [m.avg for m in train_loss_meters]
            learning_rates = [node.lr for node in self.graph]

            for model_id in range(len(self.graph)):
                if not self.graph.is_teacher(model_id):
                    logger.info(
                        "  Model %d: loss=%.4f, lr=%.6f",
                        model_id,
                        train_losses[model_id],
                        learning_rates[model_id],
                    )

            knn_scores = [0.0] * len(self.graph)
            if self.knn_eval_freq > 0 and (
                epoch % self.knn_eval_freq == 0 or epoch == epochs
            ):
                self.graph.eval_all()
                for model_id, node in enumerate(self.graph):
                    if self.graph.is_teacher(model_id):
                        continue
                    results = evaluate_knn(
                        node.model,
                        self.knn_train_loader,
                        self.knn_test_loader,
                        self.device,
                        k=self.knn_k,
                        temperature=self.knn_temperature,
                        num_classes=self.num_classes,
                    )
                    knn_scores[model_id] = results["top1"]
                    logger.info(
                        "  Model %d KNN: top1=%.2f%%", model_id, results["top1"]
                    )

            state = EpochState(
                epoch=epoch,
                train_losses=train_losses,
                train_scores=[0.0] * len(self.graph),
                val_scores=knn_scores,
                learning_rates=learning_rates,
                elapsed_time=time.time() - start_time,
            )

            for callback in self.callbacks:
                callback.on_epoch_end(self.graph, state)

            logger.info("  Elapsed time: %.2fs", state.elapsed_time)

        for callback in self.callbacks:
            callback.on_train_end(self.graph)

        logger.info("Training completed")
