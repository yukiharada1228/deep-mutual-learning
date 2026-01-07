import argparse
import os
import time

import torch
import torchvision
from models import cifar_models
from models.simclr_model import SimCLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dml.utils import AverageMeter, WorkerInitializer, save_checkpoint, set_seed
from losses import SimCLRLoss
from transform import SimCLRTransforms
from utils.optimizer import LARS
from utils.scheduler import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description="SimCLR Training on CIFAR-10")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--lr", default=0.3, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
parser.add_argument("--epochs", default=400, type=int, help="Number of epochs")
parser.add_argument("--warmup-epochs", default=10, type=int, help="Warmup epochs")
parser.add_argument("--model", default="resnet18", type=str, help="Model name")
parser.add_argument("--projection-dim", default=128, type=int, help="Projection dim")
parser.add_argument(
    "--optimizer",
    default="lars",
    type=str,
    choices=["lars", "sgd", "adam"],
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

args = parser.parse_args()
manualSeed = int(args.seed)
lr = float(args.lr)
wd = float(args.wd)
batch_size = args.batch_size
max_epoch = args.epochs
warmup_epochs = args.warmup_epochs
model_name = args.model
projection_dim = args.projection_dim
optimizer_type = args.optimizer
momentum = args.momentum
temperature = args.temperature
color_jitter_strength = args.color_jitter_strength
use_blur = args.use_blur

print("=" * 60)
print(f"SimCLR Training: {model_name}")
print("=" * 60)
print(f"Seed: {manualSeed}")
print(f"Learning rate: {lr}")
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
else:  # adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

num_training_steps = len(train_dataloader) * max_epoch
num_warmup_steps = len(train_dataloader) * warmup_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

criterion = SimCLRLoss(batch_size=batch_size, temperature=temperature)
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
            loss = criterion((z1, z2))

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

    elapsed_time = time.time() - start_time
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print()

# Close writer
writer.close()

print("=" * 60)
print("Training completed!")
print("=" * 60)
