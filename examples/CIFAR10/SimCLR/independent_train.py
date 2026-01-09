import argparse
import os
import time

import torch
import torchvision
from dml import (LARS, CompositeLoss, build_links,
                 get_cosine_schedule_with_warmup)
from dml.utils import (AverageMeter, WorkerInitializer, evaluate_knn,
                       save_checkpoint, set_seed)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from losses import SimCLRLoss
from models import cifar_models
from models.simclr_model import SimCLR
from transform import SimCLRTransforms

parser = argparse.ArgumentParser(description="SimCLR Training on CIFAR-10")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--base-lr", default=0.5, type=float, help="Base learning rate")
parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
parser.add_argument("--warmup-epochs", default=10, type=int, help="Warmup epochs")
parser.add_argument("--model", default="resnet18", type=str, help="Model name")
parser.add_argument("--projection-dim", default=128, type=int, help="Projection dim")
parser.add_argument(
    "--optimizer",
    default="lars",
    type=str,
    choices=["lars", "sgd"],
    help="Optimizer",
)
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--wd", default=1e-6, type=float, help="Weight decay")
parser.add_argument("--temperature", default=0.5, type=float, help="Temperature")
parser.add_argument(
    "--color-jitter-strength", default=0.5, type=float, help="Color jitter strength"
)
parser.add_argument(
    "--use-blur", action="store_true", help="Use Gaussian blur (not recommended)"
)
parser.add_argument(
    "--knn-eval-freq",
    type=int,
    default=1,
    help="Frequency of KNN evaluation (in epochs, 0 to disable)",
)
parser.add_argument("--knn-k", type=int, default=20, help="Number of neighbors for KNN")
parser.add_argument(
    "--knn-temperature", type=float, default=0.07, help="Temperature for KNN"
)

args = parser.parse_args()
manualSeed = int(args.seed)
base_lr = float(args.base_lr)
batch_size = args.batch_size
# Learning rate scaling: lr = base_lr * batch_size / 256
lr = base_lr * batch_size / 256
wd = float(args.wd)
max_epoch = args.epochs
warmup_epochs = args.warmup_epochs
model_name = args.model
projection_dim = args.projection_dim
optimizer_type = args.optimizer
momentum = args.momentum
temperature = args.temperature
color_jitter_strength = args.color_jitter_strength
use_blur = args.use_blur
knn_eval_freq = args.knn_eval_freq
knn_k = args.knn_k
knn_temperature = args.knn_temperature

print("=" * 60)
print(f"SimCLR Training: {model_name}")
print("=" * 60)
print(f"Seed: {manualSeed}")
print(f"Base learning rate: {base_lr}")
print(f"Learning rate (scaled): {lr}")
print(f"Weight decay: {wd}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {max_epoch}")
print(f"Warmup epochs: {warmup_epochs}")
print(f"Optimizer: {optimizer_type}")
print(f"Temperature: {temperature}")
print(f"Projection dim: {projection_dim}")
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

# Prepare the CIFAR-10 for training
num_workers = 10

train_transform = SimCLRTransforms(
    input_size=32,
    s=color_jitter_strength,
    include_blur=use_blur,
)

train_dataset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=train_transform
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

num_classes = 10

# Prepare KNN evaluation dataloaders (with standard transforms)
if knn_eval_freq > 0:

    knn_train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    knn_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    knn_train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=knn_train_transform
    )
    knn_test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=knn_test_transform
    )

    knn_train_dataloader = DataLoader(
        knn_train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    knn_test_dataloader = DataLoader(
        knn_test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

print("Setting up model...")
print()

# Setup model, optimizer, and loss function
encoder_func = lambda: getattr(cifar_models, model_name)(num_classes)
model = SimCLR(encoder_func, out_dim=projection_dim).to(device)
model_params = sum(p.numel() for p in model.parameters())
print(f"Model ({model_name}): {model_params:,} parameters")
print()

if optimizer_type == "lars":
    optimizer = LARS(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        momentum=momentum,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
        nesterov=True,
    )

num_training_steps = len(train_dataloader) * max_epoch
num_warmup_steps = len(train_dataloader) * warmup_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Use CompositeLoss from dml package
criterion_simclr = SimCLRLoss(batch_size=batch_size, temperature=temperature)
links = build_links([criterion_simclr])
criterion = CompositeLoss(links)
scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

# Setup logging and checkpointing
save_dir = f"checkpoint/independent/{model_name}"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(f"runs/independent/{model_name}")

print("=" * 60)
print("Starting training...")
print("=" * 60)
print()

# Training loop
for epoch in range(1, max_epoch + 1):
    print(f"Epoch {epoch}/{max_epoch}")
    start_time = time.time()

    # Train phase
    train_loss_meter = AverageMeter()

    model.train()
    for images, _ in train_dataloader:
        # images is a list [view1, view2] from SimCLRTransforms
        view1, view2 = images[0].to(device), images[1].to(device)

        # Forward pass
        with torch.amp.autocast(device_type=device.type):
            z1, z2 = model(view1, view2)
            # CompositeLoss expects model_id, list of outputs, list of labels, and epoch
            # For SimCLR, we pass the projection outputs as a tuple
            loss = criterion(0, [(z1, z2)], [None], epoch)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()
        scheduler.step()

        # Metrics
        train_loss_meter.update(loss.item(), view1.size(0))

    # Log training metrics
    lr_current = optimizer.param_groups[0]["lr"]
    train_loss = train_loss_meter.avg

    writer.add_scalar("train_lr", lr_current, epoch)
    writer.add_scalar("train_loss", train_loss, epoch)

    print(f"  Train: loss={train_loss:.4f}, lr={lr_current:.6f}")

    save_checkpoint(model, save_dir, epoch, filename="latest_checkpoint.pkl")

    # KNN evaluation
    if knn_eval_freq > 0 and (epoch % knn_eval_freq == 0 or epoch == max_epoch):
        print()
        print("  Running KNN evaluation...")
        results = evaluate_knn(
            model,
            knn_train_dataloader,
            knn_test_dataloader,
            device,
            k=knn_k,
            temperature=knn_temperature,
            num_classes=num_classes,
        )

        # Log KNN results
        writer.add_scalar("knn_top1", results["top1"], epoch)
        writer.add_scalar("knn_top5", results["top5"], epoch)

        print(f"  KNN: top1={results['top1']:.2f}%, top5={results['top5']:.2f}%")

    elapsed_time = time.time() - start_time
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print()

# Close writer
writer.close()

print("=" * 60)
print("Training completed!")
print("=" * 60)
