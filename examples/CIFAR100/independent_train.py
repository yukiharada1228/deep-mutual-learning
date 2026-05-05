import argparse
import logging
import os

import torch.nn as nn
from dml import (CheckpointCallback, Edge, Graph, Node, TensorBoardCallback,
                 Trainer)
from dml.utils import accuracy, set_seed
from torch.utils.tensorboard import SummaryWriter

from training_utils import (CIFAR100_NUM_CLASSES, create_cifar100_dataloaders,
                            create_grad_scaler, create_model, create_optimizer,
                            create_scheduler, get_device)


def main():
    parser = argparse.ArgumentParser(description="Independent Training on CIFAR-100")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--model", default="resnet32", type=str)

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

    model = create_model(
        model_name=args.model, device=device, num_classes=CIFAR100_NUM_CLASSES
    )
    optimizer = create_optimizer(model, lr=args.lr, wd=args.wd)
    scheduler = create_scheduler(optimizer, max_epoch=args.epochs)
    scaler = create_grad_scaler(device)

    save_dir = f"checkpoint/independent/{args.model}"
    os.makedirs(save_dir, exist_ok=True)

    graph = Graph(
        [Node(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)],
        [Edge(None, 0, nn.CrossEntropyLoss())],
    )
    Trainer(
        graph=graph,
        device=device,
        score_fn=lambda output, target: accuracy(output, target, topk=(1,))[0].item(),
        callbacks=[
            TensorBoardCallback([SummaryWriter(f"runs/independent/{args.model}")]),
            CheckpointCallback([save_dir]),
        ],
    ).fit(train_dataloader, val_dataloader, epochs=args.epochs)


if __name__ == "__main__":
    main()
