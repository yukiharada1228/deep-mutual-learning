import argparse
import logging
import os

from dml import CheckpointCallback
from dml.utils import set_seed
from losses import SimCLRLoss
from torch.utils.tensorboard import SummaryWriter

from training_utils import (
    CIFAR10_NUM_CLASSES,
    SimCLREdge,
    SimCLRGraph,
    SimCLRNode,
    SimCLRTensorBoardCallback,
    SimCLRTrainer,
    create_grad_scaler,
    create_knn_dataloaders,
    create_optimizer,
    create_scheduler,
    create_simclr_model,
    create_simclr_train_dataloader,
    get_device,
)


def main():
    parser = argparse.ArgumentParser(description="SimCLR Independent Training on CIFAR-10")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1.0, type=float)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--epochs", default=900, type=int)
    parser.add_argument("--warmup-epochs", default=10, type=int)
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--projection-dim", default=128, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-6, type=float)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--color-jitter-strength", default=0.5, type=float)
    parser.add_argument("--use-blur", action="store_true")
    parser.add_argument("--knn-eval-freq", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.07)

    args = parser.parse_args()

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

    model = create_simclr_model(args.model, device, args.projection_dim)
    optimizer = create_optimizer(model, args.lr, args.wd, args.momentum)
    num_steps = len(train_dataloader) * args.epochs
    scheduler = create_scheduler(optimizer, num_steps, len(train_dataloader) * args.warmup_epochs)
    scaler = create_grad_scaler(device)

    save_dir = f"checkpoint/independent/{args.model}"
    os.makedirs(save_dir, exist_ok=True)

    graph = SimCLRGraph(
        [SimCLRNode(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)],
        [SimCLREdge(None, 0, SimCLRLoss(batch_size=args.batch_size, temperature=args.temperature))],
    )
    SimCLRTrainer(
        graph=graph,
        device=device,
        knn_train_loader=knn_train_dataloader,
        knn_test_loader=knn_test_dataloader,
        knn_k=args.knn_k,
        knn_temperature=args.knn_temperature,
        num_classes=CIFAR10_NUM_CLASSES,
        knn_eval_freq=args.knn_eval_freq,
        callbacks=[
            SimCLRTensorBoardCallback([SummaryWriter(f"runs/independent/{args.model}")]),
            CheckpointCallback([save_dir]),
        ],
    ).fit(train_dataloader, epochs=args.epochs)


if __name__ == "__main__":
    main()
