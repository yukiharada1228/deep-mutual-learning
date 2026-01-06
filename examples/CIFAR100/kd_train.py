import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision
from dml import CompositeLoss, build_links
from dml.utils import (AverageMeter, WorkerInitializer, accuracy,
                       load_checkpoint, save_checkpoint, set_seed)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import cifar_models

parser = argparse.ArgumentParser(
    description="Knowledge Distillation (T=2) on CIFAR-100"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
parser.add_argument(
    "--temperature", default=2.0, type=float, help="Distillation temperature"
)
parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
parser.add_argument(
    "--teacher-model", default="wideresnet28_2", type=str, help="Teacher model name"
)
parser.add_argument(
    "--student-model", default="resnet32", type=str, help="Student model name"
)

args = parser.parse_args()
manualSeed = int(args.seed)
lr = float(args.lr)
wd = float(args.wd)
temperature = args.temperature
batch_size = args.batch_size
max_epoch = args.epochs
teacher_model_name = args.teacher_model
student_model_name = args.student_model

# Auto-generate teacher checkpoint path from teacher model name
teacher_checkpoint = f"checkpoint/independent/{teacher_model_name}/latest_checkpoint.pkl"
if not os.path.exists(teacher_checkpoint):
    teacher_checkpoint = None

print("=" * 60)
print(f"Knowledge Distillation with Temperature T={temperature}")
print("=" * 60)
print(f"Teacher: {teacher_model_name}")
print(f"Student: {student_model_name}")
print(f"Seed: {manualSeed}")
print(f"Learning rate: {lr}")
print(f"Weight decay: {wd}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {max_epoch}")
if teacher_checkpoint:
    print(f"Teacher checkpoint: {teacher_checkpoint}")
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

# Create teacher model
teacher = getattr(cifar_models, teacher_model_name)(num_classes).to(device)
teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"Teacher ({teacher_model_name}): {teacher_params:,} parameters")

# Load pre-trained teacher if checkpoint provided
if teacher_checkpoint:
    print(f"Loading teacher checkpoint from {teacher_checkpoint}")
    load_checkpoint(teacher, teacher_checkpoint)
    print("Teacher checkpoint loaded")
else:
    print("No teacher checkpoint provided - teacher will be trained from scratch")

# Teacher is in eval mode and frozen
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

print()

# Create student model
student = getattr(cifar_models, student_model_name)(num_classes).to(device)
student_params = sum(p.numel() for p in student.parameters())
print(f"Student ({student_model_name}): {student_params:,} parameters")
print()

# Optimizer and scheduler for student only
optimizer = torch.optim.SGD(
    student.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=wd,
    nesterov=True,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0.0
)

scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

# Setup loss functions for student
# Index 0: student (self-link)
# Index 1: teacher (cross-link from teacher to student)
criterions = [
    nn.CrossEntropyLoss(reduction="mean"),  # Student self-link (supervised)
    nn.KLDivLoss(reduction="batchmean"),  # Student learns from teacher (KD)
]

temperatures_list = [
    None,  # Student supervised: temperature not used for CrossEntropyLoss
    temperature,  # Distillation: T=2
]

links = build_links(criterions, temperatures=temperatures_list)
composite_loss = CompositeLoss(links)

print("Student loss configuration:")
for i, link in enumerate(links):
    link_type = "Self (supervised)" if i == 0 else "Teacher → Student (KD)"
    temp_str = f"{link.temperature:.1f}" if link.temperature is not None else "N/A"
    print(f"  Link {i} ({link_type}): T={temp_str}")
print()

# Setup logging and checkpointing
save_dir = (
    f"checkpoint/kd_t{temperature:.1f}/{student_model_name}_from_{teacher_model_name}"
)
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(
    f"runs/kd_t{temperature:.1f}/{student_model_name}_from_{teacher_model_name}"
)
best_score = 0.0

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
    train_score_meter = AverageMeter()

    for image, label in train_dataloader:
        image = image.to(device)
        label = label.to(device)

        # Forward pass: both student and teacher
        student.train()
        teacher.eval()  # Teacher always in eval mode

        with torch.amp.autocast(device_type=device.type):
            student_output = student(image)

        with torch.amp.autocast(device_type=device.type):
            with torch.no_grad():
                teacher_output = teacher(image)

        # Compute student loss (supervised + distillation)
        outputs = [student_output, teacher_output]
        labels = [label, label]

        with torch.amp.autocast(device_type=device.type):
            # Model ID 0 = student
            loss = composite_loss(0, outputs, labels, epoch - 1)

        # Backward pass (student only)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()

        # Metrics
        [top1] = accuracy(student_output, label, topk=(1,))
        train_score_meter.update(top1.item(), label.size(0))
        train_loss_meter.update(loss.item(), label.size(0))

    # Log training metrics
    lr_current = optimizer.param_groups[0]["lr"]
    train_loss = train_loss_meter.avg
    train_score = train_score_meter.avg

    writer.add_scalar("train_lr", lr_current, epoch)
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_score", train_score, epoch)

    print(f"  Train: loss={train_loss:.4f}, acc={train_score:.2f}%")

    scheduler.step()

    # Validation phase
    test_score_meter = AverageMeter()

    student.eval()
    for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)

        with torch.amp.autocast(device_type=device.type):
            with torch.no_grad():
                output = student(image)

        [top1] = accuracy(output, label, topk=(1,))
        test_score_meter.update(top1.item(), label.size(0))

    # Log validation metrics and save checkpoint
    test_score = test_score_meter.avg
    writer.add_scalar("test_score", test_score, epoch)

    print(f"  Test:  acc={test_score:.2f}%", end="")

    if test_score >= best_score:
        best_score = test_score
        print(" [BEST]")
    else:
        print()

    save_checkpoint(student, save_dir, epoch, filename="latest_checkpoint.pkl")

    elapsed_time = time.time() - start_time
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print()

# Close writer
writer.close()

print("=" * 60)
print("Training completed!")
print("=" * 60)
print(f"Best test accuracy (Student): {best_score:.2f}%")
print("=" * 60)
