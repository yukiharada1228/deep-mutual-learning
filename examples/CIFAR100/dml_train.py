import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

from dml import DistillationTrainer, Learner, build_links
from dml.models import cifar_models
from dml.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--wd", default=5e-4, type=float)
parser.add_argument(
    "--models",
    default=["resnet32"],
    nargs="+",
    type=str,
    help="List of model names to train with DML",
)
parser.add_argument(
    "--num-nodes",
    type=int,
    default=2,
    help="Number of nodes. If len(models) == 1, it will be broadcasted to this number.",
)

args = parser.parse_args()
manualSeed = int(args.seed)
models_name = args.models
lr = float(args.lr)
wd = float(args.wd)
num_nodes = args.num_nodes

if len(models_name) == 1 and num_nodes > 1:
    models_name = models_name * num_nodes
elif len(models_name) != num_nodes:
    raise ValueError(
        f"Length of models ({len(models_name)}) must match num_nodes ({num_nodes}) or be 1."
    )

# Fix the seed value
set_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Prepare the CIFAR-100 for training
batch_size = 64
num_workers = 10

# Normalization constants for CIFAR-100
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_dataset = torchvision.datasets.CIFAR100(
    root="data", train=True, download=True, transform=train_transform
)
val_dataset = torchvision.datasets.CIFAR100(
    root="data", train=False, download=True, transform=val_transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)

# Prepare for training
max_epoch = 200

optim_setting = {
    "name": "SGD",
    "args": {
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": wd,
        "nesterov": True,
    },
}
scheduler_setting = {
    "name": "CosineAnnealingLR",
    "args": {"T_max": max_epoch, "eta_min": 0.0},
}

num_classes = 100
learners = []
for i, model_name in enumerate(models_name):
    criterions = []
    for j in range(num_nodes):
        if i == j:
            criterions.append(nn.CrossEntropyLoss(reduction="none"))
        else:
            criterions.append(nn.KLDivLoss(reduction="none"))

    model = getattr(cifar_models, model_name)(num_classes).to(device)

    # Checkpoint path
    save_dir = f"checkpoint/dml_{num_nodes}/{i}_{model_name}"

    # Tensorboard writer
    writer = SummaryWriter(f"runs/dml_{num_nodes}/{i}_{model_name}")

    optimizer = getattr(torch.optim, optim_setting["name"])(
        model.parameters(), **optim_setting["args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
        optimizer, **scheduler_setting["args"]
    )
    links = build_links(criterions)

    learner = Learner(
        model=model,
        writer=writer,
        scaler=torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")),
        optimizer=optimizer,
        scheduler=scheduler,
        links=links,
        loss_meter=AverageMeter(),
        score_meter=AverageMeter(),
        save_dir=save_dir,
    )
    learners.append(learner)

trainer = DistillationTrainer(
    learners=learners,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
    device=device,
)
trainer.train()
