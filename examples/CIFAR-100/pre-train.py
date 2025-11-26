# Import packages
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg import KnowledgeTransferGraph, Node, build_edges
from ktg.dataset.cifar_datasets.cifar100 import get_datasets
from ktg.gates import ThroughGate
from ktg.models import cifar_models
from ktg.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--lr", default=0.1)
parser.add_argument("--wd", default=5e-4)
parser.add_argument("--model", default="resnet32")

args = parser.parse_args()
manualSeed = int(args.seed)
model_name = args.model
lr = float(args.lr)
wd = float(args.wd)

# Fix the seed value
set_seed(manualSeed)

# Prepare the CIFAR-100 for training
batch_size = 64
num_workers = 10

train_dataset, val_dataset = get_datasets()

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
nodes = []

gates_list = [ThroughGate(max_epoch)]
criterions = [nn.CrossEntropyLoss()]
model = getattr(cifar_models, model_name)(num_classes).cuda()
writer = SummaryWriter(f"runs/pre-train/{model_name}")
save_dir = f"checkpoint/pre-train/{model_name}"
optimizer = getattr(torch.optim, optim_setting["name"])(
    model.parameters(), **optim_setting["args"]
)
scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
    optimizer, **scheduler_setting["args"]
)
edges = build_edges(criterions, gates_list)

node = Node(
    model=model,
    writer=writer,
    scaler=torch.amp.GradScaler('cuda'),
    save_dir=save_dir,
    optimizer=optimizer,
    scheduler=scheduler,
    edges=edges,
    loss_meter=AverageMeter(),
    score_meter=AverageMeter(),
)
nodes.append(node)

graph = KnowledgeTransferGraph(
    nodes=nodes,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
)
graph.train()
