import argparse
import logging
import os

from torch.utils.tensorboard import SummaryWriter

from dml import (
    CheckpointCallback,
    DMLNode,
    DMLSession,
    DMLTrainer,
    TensorBoardCallback,
    build_mutual_learning_losses,
)
from dml.utils import accuracy, set_seed
from training_utils import (
    CIFAR100_NUM_CLASSES,
    create_cifar100_dataloaders,
    create_grad_scaler,
    create_model,
    create_optimizer,
    create_scheduler,
    get_device,
)


def main():
    parser = argparse.ArgumentParser(description="Independent Training on CIFAR-100")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--model", default="resnet32", type=str, help="Model name")

    args = parser.parse_args()
    manualSeed = int(args.seed)
    lr = float(args.lr)
    wd = float(args.wd)
    batch_size = args.batch_size
    max_epoch = args.epochs
    model_name = args.model

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Independent Training: %s", model_name)
    logger.info("=" * 60)
    logger.info("Seed: %d", manualSeed)
    logger.info("Learning rate: %g", lr)
    logger.info("Weight decay: %g", wd)
    logger.info("Batch size: %d", batch_size)
    logger.info("Epochs: %d", max_epoch)
    logger.info("=" * 60)

    set_seed(manualSeed)

    device = get_device()
    logger.info("Using device: %s", device)

    train_dataloader, val_dataloader = create_cifar100_dataloaders(
        batch_size=batch_size,
        seed=manualSeed,
    )

    logger.info("Setting up model...")

    losses = build_mutual_learning_losses(num_nodes=1)

    model, model_params = create_model(
        model_name=model_name,
        device=device,
        num_classes=CIFAR100_NUM_CLASSES,
    )
    logger.info("Model (%s): %s parameters", model_name, f"{model_params:,}")

    optimizer = create_optimizer(model, lr=lr, wd=wd)
    scheduler = create_scheduler(optimizer, max_epoch=max_epoch)
    scaler = create_grad_scaler(device)

    save_dir = f"checkpoint/independent/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    node = DMLNode(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    session = DMLSession([node], losses)

    writer = SummaryWriter(f"runs/independent/{model_name}")
    callbacks = [
        TensorBoardCallback([writer]),
        CheckpointCallback([save_dir]),
    ]
    trainer = DMLTrainer(
        session=session,
        device=device,
        score_fn=lambda output, target: accuracy(output, target, topk=(1,))[0].item(),
        callbacks=callbacks,
    )

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    trainer.fit(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=max_epoch,
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
