import argparse
import logging
import os

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dml import (CheckpointCallback, Edge, Graph, KLLoss, Node,
                 TensorBoardCallback, Trainer, accuracy,
                 create_classification_evaluator)
from dml.utils import create_grad_scaler, get_device, set_seed

from .training_utils import (CIFAR100_NUM_CLASSES, create_cifar100_dataloaders,
                             create_model, create_optimizer, create_scheduler)


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
        "--students",
        default=["resnet32"],
        nargs="+",
        type=str,
        help="Student model name(s) to train",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    device = get_device()

    train_dataloader, val_dataloader = create_cifar100_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    score_fn = lambda output, target: accuracy(output, target, topk=(1,))[0].item()
    eval_fn = create_classification_evaluator(val_dataloader, score_fn)

    teacher_nodes = []
    for name in args.teachers:
        model = create_model(name, device, CIFAR100_NUM_CLASSES)
        ckpt_path = os.path.join(
            "checkpoint", "independent", name, "latest_checkpoint.pth"
        )
        teacher_nodes.append(Node(model=model, checkpoint_path=ckpt_path))

    student_nodes, student_writers, student_save_dirs = [], [], []
    for i, name in enumerate(args.students):
        model = create_model(name, device, CIFAR100_NUM_CLASSES)
        optimizer = create_optimizer(model, lr=args.lr, wd=args.wd)
        scheduler = create_scheduler(optimizer, max_epoch=args.epochs)
        scaler = create_grad_scaler(device)
        save_dir = f"checkpoint/kd_t{args.temperature:.1f}/{i}_{name}"
        os.makedirs(save_dir, exist_ok=True)
        student_nodes.append(
            Node(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                eval_fn=eval_fn,
            )
        )
        student_writers.append(
            SummaryWriter(f"runs/kd_t{args.temperature:.1f}/{i}_{name}")
        )
        student_save_dirs.append(save_dir)

    nodes = student_nodes + teacher_nodes
    student_idx = range(len(student_nodes))
    teacher_idx = range(len(student_nodes), len(nodes))

    graph = Graph(
        nodes,
        [
            *[Edge(None, s, nn.CrossEntropyLoss()) for s in student_idx],
            *[
                Edge(
                    t,
                    s,
                    KLLoss(temperature=args.temperature),
                )
                for t in teacher_idx
                for s in student_idx
            ],
        ],
    )
    Trainer(
        graph=graph,
        device=device,
        callbacks=[
            TensorBoardCallback(student_writers + [None] * len(teacher_nodes)),
            CheckpointCallback(student_save_dirs + [None] * len(teacher_nodes)),
        ],
    ).fit(train_dataloader, epochs=args.epochs)


if __name__ == "__main__":
    main()
