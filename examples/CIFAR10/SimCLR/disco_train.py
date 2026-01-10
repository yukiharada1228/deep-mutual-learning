import argparse
import os
import time

import torch
import torchvision
from losses import DisCOLoss, SimCLRLoss
from models import cifar_models
from models.simclr_model import SimCLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transform import SimCLRTransforms

from dml import (LARS, CompositeLoss, build_links,
                 get_cosine_schedule_with_warmup)
from dml.utils import (AverageMeter, WorkerInitializer, evaluate_knn,
                       load_checkpoint, save_checkpoint, set_seed)


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR + DisCO (Teacher-Student) on CIFAR-10"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--base-lr", default=0.5, type=float, help="Base learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--warmup-epochs", default=10, type=int, help="Warmup epochs")
    parser.add_argument(
        "--projection-dim", default=128, type=int, help="Projection dim"
    )
    parser.add_argument(
        "--optimizer",
        default="lars",
        type=str,
        choices=["lars", "sgd"],
        help="Optimizer",
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--wd", default=1e-6, type=float, help="Weight decay")
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="Temperature for SimCLR"
    )
    parser.add_argument(
        "--color-jitter-strength", default=0.5, type=float, help="Color jitter strength"
    )
    parser.add_argument(
        "--use-blur", action="store_true", help="Use Gaussian blur (not recommended)"
    )

    # Model arguments
    parser.add_argument(
        "--teacher-model",
        default="resnet50",
        type=str,
        help="Teacher model architecture",
    )
    parser.add_argument(
        "--student-model",
        default="resnet18",
        type=str,
        help="Student model architecture",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        type=str,
        help="Path to teacher checkpoint",
    )

    # Loss weight argument
    parser.add_argument(
        "--weight-disco", default=1.0, type=float, help="Weight for DisCO loss"
    )

    parser.add_argument(
        "--knn-eval-freq",
        type=int,
        default=1,
        help="Frequency of KNN evaluation (in epochs, 0 to disable)",
    )
    parser.add_argument(
        "--knn-k", type=int, default=20, help="Number of neighbors for KNN"
    )
    parser.add_argument(
        "--knn-temperature", type=float, default=0.07, help="Temperature for KNN"
    )

    args = parser.parse_args()
    manualSeed = int(args.seed)

    # Args extraction
    base_lr = float(args.base_lr)
    batch_size = args.batch_size
    lr = base_lr * batch_size / 256
    wd = float(args.wd)
    max_epoch = args.epochs
    warmup_epochs = args.warmup_epochs
    projection_dim = args.projection_dim
    optimizer_type = args.optimizer
    momentum = args.momentum
    temperature = args.temperature
    color_jitter_strength = args.color_jitter_strength
    use_blur = args.use_blur

    teacher_model_name = args.teacher_model
    student_model_name = args.student_model
    teacher_checkpoint = args.teacher_checkpoint
    weight_disco = args.weight_disco

    knn_eval_freq = args.knn_eval_freq
    knn_k = args.knn_k
    knn_temperature = args.knn_temperature

    print("=" * 60)
    print(f"SimCLR + DisCO Training (Teacher-Student)")
    print("=" * 60)
    print(f"Teacher: {teacher_model_name} (from {teacher_checkpoint})")
    print(f"Student: {student_model_name}")
    print(f"Seed: {manualSeed}")
    print(f"Base learning rate: {base_lr}")
    print(f"Learning rate (scaled): {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {max_epoch}")
    print(f"Weight DisCO: {weight_disco}")
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

    # Prepare datasets
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

    # KNN evaluation setup
    if knn_eval_freq > 0:
        knn_train_transform = transforms.Compose([transforms.ToTensor()])
        knn_test_transform = transforms.Compose([transforms.ToTensor()])
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
        )
        knn_test_dataloader = DataLoader(
            knn_test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print("Setting up models...")

    # 1. Setup Student
    encoder_func_s = lambda name=student_model_name: getattr(cifar_models, name)(
        num_classes
    )
    student_model = SimCLR(encoder_func_s, out_dim=projection_dim).to(device)
    print(
        f"Student ({student_model_name}): {sum(p.numel() for p in student_model.parameters()):,} params"
    )

    # 2. Setup Teacher
    encoder_func_t = lambda name=teacher_model_name: getattr(cifar_models, name)(
        num_classes
    )
    teacher_model = SimCLR(encoder_func_t, out_dim=projection_dim).to(device)
    print(
        f"Teacher ({teacher_model_name}): {sum(p.numel() for p in teacher_model.parameters()):,} params"
    )

    # Load teacher weights and freeze
    load_checkpoint(teacher_model, teacher_checkpoint)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen.")
    print()

    # Optimization
    if optimizer_type == "lars":
        optimizer = LARS(
            student_model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            weight_decay_filter=True,
            lars_adaptation_filter=True,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            student_model.parameters(),
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

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Loss Functions via dml package
    # Student node: 0, Teacher node: 1
    # We construct CompositeLoss for Student (0)
    # Incoming links for Student:
    # - From Student (0): SimCLRLoss
    # - From Teacher (1): DisCOLoss

    criterions = []
    # Link 0: Self (SimCLR)
    criterions.append(SimCLRLoss(batch_size=batch_size, temperature=temperature))
    # Link 1: From Teacher (DisCO)
    criterions.append(DisCOLoss())

    links = build_links(criterions)
    composite_loss = CompositeLoss(links)

    # Logging
    save_dir = f"checkpoint/disco/{student_model_name}_distill_{teacher_model_name}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(
        f"runs/disco/{student_model_name}_distill_{teacher_model_name}"
    )

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    for epoch in range(1, max_epoch + 1):
        print(f"Epoch {epoch}/{max_epoch}")
        start_time = time.time()

        student_model.train()
        loss_meter = AverageMeter()

        for images, _ in train_dataloader:
            view1, view2 = images[0].to(device), images[1].to(device)

            # Helper for forward pass
            def forward_step(model, v1, v2):
                with torch.amp.autocast(device_type=device.type):
                    return model(v1, v2)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_z1, t_z2 = forward_step(teacher_model, view1, view2)

            # Student forward
            with torch.amp.autocast(device_type=device.type):
                s_z1, s_z2 = student_model(view1, view2)

                # Prepare outputs and labels for CompositeLoss
                # outputs: [student_output, teacher_output]
                outputs = [(s_z1, s_z2), (t_z1, t_z2)]
                labels = [None, None]  # Not used for SimCLR/DisCO

                # Compute loss for Student (Node 0)
                loss = composite_loss(0, outputs, labels)

            # Optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            scheduler.step()
            # Update metrics
            loss_meter.update(loss.item(), view1.size(0))

        # Log metrics
        lr_current = optimizer.param_groups[0]["lr"]

        # Match dml_train.py keys
        writer.add_scalar("train_lr", lr_current, epoch)
        writer.add_scalar("train_loss", loss_meter.avg, epoch)

        # Match dml_train.py print format (Student is implicitly Model 0)
        print(f"  Model 0: loss={loss_meter.avg:.4f}, lr={lr_current:.6f}")

        save_checkpoint(
            student_model,
            save_dir,
            epoch,
            filename="latest_checkpoint.pkl",
        )

        # KNN Evaluation
        if knn_eval_freq > 0 and (epoch % knn_eval_freq == 0 or epoch == max_epoch):
            print()
            print("  Running KNN evaluation...")
            results = evaluate_knn(
                student_model,
                knn_train_dataloader,
                knn_test_dataloader,
                device,
                k=knn_k,
                temperature=knn_temperature,
                num_classes=num_classes,
            )
            # Match dml_train.py keys
            writer.add_scalar("knn_top1", results["top1"], epoch)
            writer.add_scalar("knn_top5", results["top5"], epoch)

            # Match dml_train.py print format
            print(
                f"  Model 0 KNN: top1={results['top1']:.2f}%, top5={results['top5']:.2f}%"
            )

        elapsed_time = time.time() - start_time
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print()

    writer.close()

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
