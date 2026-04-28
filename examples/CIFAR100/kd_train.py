import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from training_utils import (CIFAR100_NUM_CLASSES, create_cifar100_dataloaders,
                            create_grad_scaler, create_model, create_optimizer,
                            create_scheduler, get_device)

from dml import (CheckpointCallback, Edge, Node, Graph, TensorBoardCallback,
                 Trainer)
from dml.utils import accuracy, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation (KD) on CIFAR-100"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--temperature", default=2.0, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--teachers",
        default=["wideresnet28_2"],
        nargs="+",
        type=str,
        help="Frozen teacher model name(s)",
    )
    parser.add_argument(
        "--teacher-checkpoints",
        default=[],
        nargs="+",
        type=str,
        help="Checkpoint paths for each teacher (must match --teachers length)",
    )
    parser.add_argument(
        "--students",
        default=["resnet32"],
        nargs="+",
        type=str,
        help="Student model name(s) to train",
    )

    args = parser.parse_args()

    if args.teacher_checkpoints and len(args.teacher_checkpoints) != len(args.teachers):
        raise ValueError(
            "--teacher-checkpoints must have the same length as --teachers"
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    device = get_device()

    train_dataloader, val_dataloader = create_cifar100_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    num_teachers = len(args.teachers)
    num_students = len(args.students)

    nodes = []
    writers = []
    save_dirs = []

    for i, name in enumerate(args.teachers):
        model = create_model(name, device, CIFAR100_NUM_CLASSES)
        if args.teacher_checkpoints:
            ckpt = torch.load(args.teacher_checkpoints[i], map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))

        nodes.append(Node(model=model))
        writers.append(
            SummaryWriter(f"runs/kd_t{args.temperature:.1f}/teacher_{i}_{name}")
        )
        save_dirs.append(None)

    for i, name in enumerate(args.students):
        model = create_model(name, device, CIFAR100_NUM_CLASSES)
        optimizer = create_optimizer(model, lr=args.lr, wd=args.wd)
        scheduler = create_scheduler(optimizer, max_epoch=args.epochs)
        scaler = create_grad_scaler(device)

        save_dir = f"checkpoint/kd_t{args.temperature:.1f}/{i}_{name}"
        os.makedirs(save_dir, exist_ok=True)

        nodes.append(
            Node(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        )
        writers.append(SummaryWriter(f"runs/kd_t{args.temperature:.1f}/{i}_{name}"))
        save_dirs.append(save_dir)

    session = Graph(
        nodes,
        [
            *[Edge(None, num_teachers + i, nn.CrossEntropyLoss()) for i in range(num_students)],
            *[
                Edge(t, num_teachers + s, nn.KLDivLoss(reduction="batchmean"), args.temperature)
                for t in range(num_teachers)
                for s in range(num_students)
            ],
        ],
    )
    Trainer(
        session=session,
        device=device,
        score_fn=lambda output, target: accuracy(output, target, topk=(1,))[0].item(),
        callbacks=[
            TensorBoardCallback(writers),
            CheckpointCallback(save_dirs),
        ],
    ).fit(train_dataloader, val_dataloader, epochs=args.epochs)


if __name__ == "__main__":
    main()
