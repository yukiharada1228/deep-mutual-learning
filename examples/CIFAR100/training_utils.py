import torch
import torchvision
from dml.utils import WorkerInitializer
from torch.utils.data import DataLoader
from torchvision import transforms

from models import cifar_models

CIFAR100_NUM_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
DEFAULT_NUM_WORKERS = 10


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_cifar100_dataloaders(
    batch_size: int,
    seed: int,
    data_root: str = "data",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
    )

    worker_initializer = WorkerInitializer(seed).worker_init_fn
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": worker_initializer,
    }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )
    return train_dataloader, val_dataloader


def create_model(
    model_name: str,
    device: torch.device,
    num_classes: int = CIFAR100_NUM_CLASSES,
):
    return getattr(cifar_models, model_name)(num_classes).to(device)


def create_optimizer(model: torch.nn.Module, lr: float, wd: float):
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
        nesterov=True,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, max_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,
        eta_min=0.0,
    )


def create_grad_scaler(device: torch.device):
    return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
