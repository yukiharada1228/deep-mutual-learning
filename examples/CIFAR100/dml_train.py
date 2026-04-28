import argparse
import logging
import os

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from training_utils import (CIFAR100_NUM_CLASSES, create_cifar100_dataloaders,
                            create_grad_scaler, create_model, create_optimizer,
                            create_scheduler, get_device)

from dml import (CheckpointCallback, Edge, Node, Session, TensorBoardCallback,
                 Trainer)
from dml.utils import accuracy, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Deep Mutual Learning (DML) on CIFAR-100"
    )
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
    models_name = args.models
    temperature = args.temperature
    num_nodes = args.num_nodes

    if len(models_name) == 1 and num_nodes > 1:
        models_name = models_name * num_nodes
    elif len(models_name) != num_nodes:
        raise ValueError(
            f"Length of models ({len(models_name)}) must match num_nodes ({num_nodes}) or be 1."
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

    nodes = []
    writers = []
    save_dirs = []

    for i, model_name in enumerate(models_name):
        model, _ = create_model(
            model_name=model_name, device=device, num_classes=CIFAR100_NUM_CLASSES
        )
        optimizer = create_optimizer(model, lr=args.lr, wd=args.wd)
        scheduler = create_scheduler(optimizer, max_epoch=args.epochs)
        scaler = create_grad_scaler(device)

        save_dir = f"checkpoint/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

        writers.append(
            SummaryWriter(f"runs/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}")
        )
        nodes.append(
            Node(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        )

    session = Session(
        nodes,
        *(Edge(None, i, nn.CrossEntropyLoss()) for i in range(num_nodes)),
        *(
            Edge(src, tgt, nn.KLDivLoss(reduction="batchmean"), temperature)
            for src in range(num_nodes)
            for tgt in range(num_nodes)
            if src != tgt
        ),
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
