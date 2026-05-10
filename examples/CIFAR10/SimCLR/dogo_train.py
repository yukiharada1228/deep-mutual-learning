"""SimCLR + Distill on the Go (DoGo) on CIFAR-10.

Two or more models learn collaboratively in a self-supervised setting:
  - Each model is trained with the NT-Xent (SimCLR) self-supervised loss.
  - Each model additionally distills from every peer by aligning their pairwise
    cross-view similarity structure via KL divergence (DoGo loss).

This implements online mutual distillation without a pre-trained teacher.

Reference:
    Bhat et al. "Distill on the Go: Online knowledge distillation in
    self-supervised learning." CVPR Workshops 2021.
    https://arxiv.org/abs/2104.09866
"""

import argparse
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dml import (CheckpointCallback, Edge, Graph, Node, TensorBoardCallback,
                 Trainer)
from dml.utils import create_grad_scaler, get_device, set_seed

from .knn_eval import create_knn_evaluator
from .losses import DoGoLoss, NTXentLoss
from .training_utils import (CIFAR10_NUM_CLASSES, contrastive_model_inputs,
                             create_knn_dataloaders, create_optimizer,
                             create_scheduler, create_simclr_model,
                             create_simclr_train_dataloader,
                             prepare_contrastive_batch)


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR + DoGo (Online Mutual Relational Distillation) on CIFAR-10"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--lr",
        default=1.0,
        type=float,
        help="Learning rate (default: 1.0 for LARS)",
    )
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size (default: 512)"
    )
    parser.add_argument("--epochs", default=900, type=int)
    parser.add_argument(
        "--warmup-epochs",
        default=10,
        type=int,
        help="Warmup epochs (default: 10)",
    )
    parser.add_argument(
        "--models",
        default=["resnet50"],
        nargs="+",
        type=str,
        help="List of model names to train with DoGo",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=2,
        help="Number of nodes. If --models has 1 entry, it is replicated this many times.",
    )
    parser.add_argument("--projection-dim", default=128, type=int)
    parser.add_argument("--num-proj-layers", default=2, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-6, type=float)
    parser.add_argument(
        "--disable-scheduler",
        action="store_true",
        help="Disable cosine scheduler",
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature for NTXentLoss (contrastive loss)",
    )
    parser.add_argument(
        "--dogo-temperature",
        default=0.1,
        type=float,
        help="Temperature for DoGoLoss (distillation loss)",
    )
    parser.add_argument("--color-jitter-strength", default=0.5, type=float)
    parser.add_argument("--use-blur", action="store_true")
    parser.add_argument(
        "--dogo-distill-weight",
        default=1.0,
        type=float,
        help=(
            "Loss weight for DoGo distillation. "
            "This implementation uses reduction='batchmean' and applies T^2 scaling "
            "to ensure gradient magnitude consistency across temperatures."
        ),
    )
    parser.add_argument("--knn-eval-freq", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.07)

    args = parser.parse_args()

    if len(args.models) == 1:
        model_names = args.models * args.num_nodes
    elif len(args.models) == args.num_nodes:
        model_names = args.models
    else:
        raise ValueError(
            f"--models length ({len(args.models)}) must equal --num-nodes ({args.num_nodes}) or be 1."
        )

    num_nodes = len(model_names)
    temperature = args.temperature

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    device = get_device()

    train_dataloader = create_simclr_train_dataloader(
        batch_size=args.batch_size,
        seed=args.seed,
        color_jitter_strength=args.color_jitter_strength,
        include_blur=args.use_blur,
    )
    knn_train_dataloader, knn_test_dataloader = create_knn_dataloaders()

    num_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = len(train_dataloader) * args.warmup_epochs

    nodes, writers, save_dirs = [], [], []
    for i, name in enumerate(model_names):
        model = create_simclr_model(
            name, device, args.projection_dim, args.num_proj_layers
        )
        optimizer = create_optimizer(model, args.lr, args.wd, args.momentum)
        if args.disable_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: 1.0
            )
        else:
            scheduler = create_scheduler(optimizer, num_steps, num_warmup_steps)
        scaler = create_grad_scaler(device)

        knn_eval = create_knn_evaluator(
            knn_train_dataloader,
            knn_test_dataloader,
            k=args.knn_k,
            temperature=args.knn_temperature,
            num_classes=CIFAR10_NUM_CLASSES,
            eval_freq=args.knn_eval_freq,
        )

        save_dir = f"checkpoint/dogo_t{temperature:.1f}_n{num_nodes}/{i}_{name}"
        os.makedirs(save_dir, exist_ok=True)

        nodes.append(
            Node(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                eval_fn=knn_eval,
                scheduler_interval="step",
                model_input_fn=contrastive_model_inputs,
            )
        )
        writers.append(
            SummaryWriter(f"runs/dogo_t{temperature:.1f}_n{num_nodes}/{i}_{name}")
        )
        save_dirs.append(save_dir)

    graph = Graph(
        nodes,
        [
            # Self-supervised (SimCLR) losses — one per node
            *[
                Edge(
                    None,
                    i,
                    NTXentLoss(batch_size=args.batch_size, temperature=temperature),
                )
                for i in range(num_nodes)
            ],
            # Mutual distillation (DoGo) losses — all peer pairs
            *[
                Edge(
                    src,
                    tgt,
                    DoGoLoss(temperature=args.dogo_temperature),
                    weight=args.dogo_distill_weight,
                )
                for src in range(num_nodes)
                for tgt in range(num_nodes)
                if src != tgt
            ],
        ],
    )
    Trainer(
        graph=graph,
        device=device,
        prepare_batch=prepare_contrastive_batch(device),
        callbacks=[
            TensorBoardCallback(writers),
            CheckpointCallback(save_dirs),
        ],
    ).fit(train_dataloader, epochs=args.epochs)


if __name__ == "__main__":
    main()
