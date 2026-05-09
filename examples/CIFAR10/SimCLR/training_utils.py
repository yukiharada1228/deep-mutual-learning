import torch
import torchvision
from cosine_warmup import get_cosine_schedule_with_warmup
from lars import LARS
from models import cifar_models
from models.simclr_model import SimCLR
from torch.utils.data import DataLoader
from torchvision import transforms
from transform import SimCLRTransforms

from dml.utils import WorkerInitializer

CIFAR10_NUM_CLASSES = 10
DEFAULT_NUM_WORKERS = 10


def prepare_contrastive_batch(device: torch.device):
    """Create a batch preparation function for contrastive learning."""

    def prepare_batch(raw_batch) -> dict:
        images, _ = raw_batch
        return {"views": [v.to(device) for v in images]}

    return prepare_batch


def contrastive_model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a contrastive batch to the two model inputs used by SimCLR."""
    return batch["views"][0], batch["views"][1]


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
