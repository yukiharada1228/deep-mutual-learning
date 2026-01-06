import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision
from models import cifar_models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dml import CompositeLoss, build_links
from dml.utils import (AverageMeter, WorkerInitializer, accuracy,
                       save_checkpoint, set_seed)

parser = argparse.ArgumentParser(description="Deep Mutual Learning (DML) on CIFAR-100")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
parser.add_argument(
    "--temperature", default=1.0, type=float, help="Distillation temperature"
)
parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
parser.add_argument(
    "--models",
    default=["resnet32", "wideresnet28_2"],
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
temperature = args.temperature
batch_size = args.batch_size
max_epoch = args.epochs
num_nodes = args.num_nodes

if len(models_name) == 1 and num_nodes > 1:
    models_name = models_name * num_nodes
elif len(models_name) != num_nodes:
    raise ValueError(
        f"Length of models ({len(models_name)}) must match num_nodes ({num_nodes}) or be 1."
    )

print("=" * 60)
print(f"Deep Mutual Learning with Temperature T={temperature}")
print("=" * 60)
print(f"Models: {models_name}")
print(f"Seed: {manualSeed}")
print(f"Learning rate: {lr}")
print(f"Weight decay: {wd}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {max_epoch}")
print("=" * 60)
print()

# Fix the seed value
set_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print()

# Prepare the CIFAR-100 for training
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

num_classes = 100

print("Setting up models...")
print()

# Setup models, optimizers, and loss functions
models = []
optimizers = []
schedulers = []
scalers = []
composite_losses = []
writers = []
save_dirs = []
best_scores = []

for i, model_name in enumerate(models_name):
    # Create model
    model = getattr(cifar_models, model_name)(num_classes).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model {i} ({model_name}): {model_params:,} parameters")
    models.append(model)

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
        nesterov=True,
    )
    optimizers.append(optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=0.0
    )
    schedulers.append(scheduler)

    # Create scaler for mixed precision
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    scalers.append(scaler)

    # Create loss functions (criterions) for this model
    criterions = []
    temperatures_list = []

    for j in range(num_nodes):
        if i == j:
            # Self-link: supervised learning
            criterions.append(nn.CrossEntropyLoss(reduction="mean"))
            temperatures_list.append(None)
        else:
            # Cross-link: knowledge distillation
            criterions.append(nn.KLDivLoss(reduction="batchmean"))
            temperatures_list.append(temperature)

    links = build_links(criterions, temperatures=temperatures_list)
    composite_loss = CompositeLoss(links)
    composite_losses.append(composite_loss)

    # Print link configuration
    print(f"  Loss config for Model {i}:")
    for k, link in enumerate(links):
        link_type = "Self (supervised)" if i == k else f"Node {k} → Node {i} (KD)"
        temp_str = f"{link.temperature:.1f}" if link.temperature is not None else "N/A"
        print(f"    Link {k} ({link_type}): T={temp_str}")

    # Setup logging and checkpointing
    # e.g. checkpoint/dml_t2.0_n4/0_resnet32
    save_dir = f"checkpoint/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_dirs.append(save_dir)

    writer = SummaryWriter(f"runs/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}")
    writers.append(writer)

    best_scores.append(0.0)

print()
print("=" * 60)
print("Starting training...")
print("=" * 60)
print()

# Training loop
for epoch in range(1, max_epoch + 1):
    print(f"Epoch {epoch}/{max_epoch}")
    start_time = time.time()

    # Train phase
    train_loss_meters = [AverageMeter() for _ in range(num_nodes)]
    train_score_meters = [AverageMeter() for _ in range(num_nodes)]

    for image, label in train_dataloader:
        image = image.to(device)
        label = label.to(device)

        # Forward pass for all models
        outputs = []
        for model in models:
            model.train()
            with torch.amp.autocast(device_type=device.type):
                output = model(image)
            outputs.append(output)

        # Backward pass for each model
        labels = [label] * num_nodes
        for model_id in range(num_nodes):
            with torch.amp.autocast(device_type=device.type):
                loss = composite_losses[model_id](model_id, outputs, labels, epoch - 1)

            # Optimization
            scalers[model_id].scale(loss).backward()
            scalers[model_id].step(optimizers[model_id])
            optimizers[model_id].zero_grad()
            scalers[model_id].update()

            # Metrics
            [top1] = accuracy(outputs[model_id], labels[model_id], topk=(1,))
            train_score_meters[model_id].update(top1.item(), label.size(0))
            train_loss_meters[model_id].update(loss.item(), label.size(0))

    # Log training metrics
    for model_id in range(num_nodes):
        lr_current = optimizers[model_id].param_groups[0]["lr"]
        train_loss = train_loss_meters[model_id].avg
        train_score = train_score_meters[model_id].avg

        writers[model_id].add_scalar("train_lr", lr_current, epoch)
        writers[model_id].add_scalar("train_loss", train_loss, epoch)
        writers[model_id].add_scalar("train_score", train_score, epoch)

        print(f"  Model {model_id}: loss={train_loss:.4f}, acc={train_score:.2f}%")

        schedulers[model_id].step()

    # Validation phase
    test_score_meters = [AverageMeter() for _ in range(num_nodes)]

    for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)

        # Forward pass for all models
        for model_id, model in enumerate(models):
            model.eval()
            with torch.amp.autocast(device_type=device.type):
                with torch.no_grad():
                    output = model(image)

            [top1] = accuracy(output, label, topk=(1,))
            test_score_meters[model_id].update(top1.item(), label.size(0))

    # Log validation metrics and save checkpoints
    for model_id in range(num_nodes):
        test_score = test_score_meters[model_id].avg
        writers[model_id].add_scalar("test_score", test_score, epoch)

        print(f"  Model {model_id}: test_acc={test_score:.2f}%", end="")

        if test_score >= best_scores[model_id]:
            best_scores[model_id] = test_score
            print(" [BEST]")
        else:
            print()

        save_checkpoint(
            models[model_id],
            save_dirs[model_id],
            epoch,
            filename="latest_checkpoint.pkl",
        )

    elapsed_time = time.time() - start_time
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print()

# Close writers
for writer in writers:
    writer.close()

print("=" * 60)
print("Training completed!")
print("=" * 60)
for i, score in enumerate(best_scores):
    print(f"Model {i} best test accuracy: {score:.2f}%")
print("=" * 60)
