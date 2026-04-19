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
        default=["resnet20"],
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
    manualSeed = int(args.seed)
    models_name = args.models
    lr = float(args.lr)
    wd = float(args.wd)
    temperature = args.temperature
    batch_size = args.batch_size
    max_epoch = args.epochs
    num_nodes = args.num_nodes

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    if len(models_name) == 1 and num_nodes > 1:
        models_name = models_name * num_nodes
    elif len(models_name) != num_nodes:
        raise ValueError(
            f"Length of models ({len(models_name)}) must match num_nodes ({num_nodes}) or be 1."
        )

    logger.info("=" * 60)
    logger.info("Deep Mutual Learning with Temperature T=%g", temperature)
    logger.info("=" * 60)
    logger.info("Models: %s", models_name)
    logger.info("Seed: %d", manualSeed)
    logger.info("Learning rate: %g", lr)
    logger.info("Weight decay: %g", wd)
    logger.info("Batch size: %d", batch_size)
    logger.info("Epochs: %d", max_epoch)
    logger.info("=" * 60)

    # Fix the seed value
    set_seed(manualSeed)

    device = get_device()
    logger.info("Using device: %s", device)

    train_dataloader, val_dataloader = create_cifar100_dataloaders(
        batch_size=batch_size,
        seed=manualSeed,
    )

    logger.info("Setting up models...")

    losses = build_mutual_learning_losses(
        num_nodes=num_nodes,
        temperature=temperature,
    )
    nodes = []
    writers = []
    save_dirs = []

    for i, model_name in enumerate(models_name):
        model, model_params = create_model(
            model_name=model_name,
            device=device,
            num_classes=CIFAR100_NUM_CLASSES,
        )
        logger.info("Model %d (%s): %s parameters", i, model_name, f"{model_params:,}")

        logger.info("  Loss config for Model %d:", i)
        for k, link in enumerate(losses[i].incoming_links):
            link_type = "Self (supervised)" if i == k else f"Node {k} → Node {i} (KD)"
            temp_str = (
                f"{link.temperature:.1f}" if link.temperature is not None else "N/A"
            )
            logger.info("    Link %d (%s): T=%s", k, link_type, temp_str)

        optimizer = create_optimizer(model, lr=lr, wd=wd)
        scheduler = create_scheduler(optimizer, max_epoch=max_epoch)
        scaler = create_grad_scaler(device)

        save_dir = f"checkpoint/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

        writer = SummaryWriter(
            f"runs/dml_t{temperature:.1f}_n{num_nodes}/{i}_{model_name}"
        )
        writers.append(writer)
        nodes.append(
            DMLNode(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        )

    session = DMLSession(nodes, losses)
    callbacks = [
        TensorBoardCallback(writers),
        CheckpointCallback(save_dirs),
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
