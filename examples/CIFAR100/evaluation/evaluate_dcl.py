import argparse
import os
from typing import List, Optional

import optuna
import torch
import torch.nn as nn
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcl import DistillationTrainer, Learner, build_links, gates
from dcl.dataset.cifar_datasets.cifar100 import get_datasets
from dcl.models import cifar_models
from dcl.utils import (AverageMeter, WorkerInitializer, load_checkpoint,
                       set_seed)


def infer_model_names(
    best_trial: optuna.trial.FrozenTrial, num_nodes: int
) -> List[str]:
    """
    Restore model names for each node from Optuna trial information.
    """
    model_names: List[Optional[str]] = [None] * num_nodes

    # 1) Fill in from params (node 1 onwards)
    for i in range(1, num_nodes):
        if not model_names[i]:
            key = f"{i}_model"
            val = best_trial.params.get(key)
            if isinstance(val, str) and len(val) > 0:
                model_names[i] = val

    # 2) Fallback for node 0 (training default: models[0] = resnet32)
    if not model_names[0]:
        model_names[0] = "resnet32"

    if any(m is None or len(m) == 0 for m in model_names):
        missing = [i for i, m in enumerate(model_names) if not m]
        raise RuntimeError(
            f"Could not identify model names. trial={best_trial.number}, missing nodes={missing}"
        )
    return [str(m) for m in model_names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--trial", type=int, default=None, help="Trial number to fix")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name. If not specified, dcl_{num_nodes} is used",
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study_name = args.study_name or f"dcl_{args.num_nodes}"
    optuna_dir = os.path.join("../optuna", study_name)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    study = optuna.create_study(
        storage=storage, study_name=study_name, load_if_exists=True
    )

    if args.trial is not None:
        # Use specified trial (existence check)
        frozen = None
        for t in study.trials:
            if t.number == args.trial:
                frozen = t
                break
        if frozen is None:
            raise ValueError(f"Specified trial not found: {args.trial}")
        best_trial = frozen
    else:
        # Use best trial
        if study.best_trial is None:
            raise RuntimeError("best_trial not found. Training may be incomplete.")
        best_trial = study.best_trial

    model_names = infer_model_names(best_trial, args.num_nodes)

    # Always retrain: Rebuild graph with best trial configuration and train→test
    # use_test_mode=True: get train(=train+val) and test
    train_dataset, test_dataset = get_datasets(use_test_mode=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=WorkerInitializer(args.seed).worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=WorkerInitializer(args.seed).worker_init_fn,
    )

    max_epoch = int(args.max_epoch)
    # Training settings (same as dcl_train.py)
    optim_setting = {
        "name": "SGD",
        "args": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
    }
    scheduler_setting = {
        "name": "CosineAnnealingLR",
        "args": {"T_max": max_epoch, "eta_min": 0.0},
    }

    num_classes = 100
    learners: List[Learner] = []
    # Build learners
    for i in range(args.num_nodes):
        # Gates and losses
        gates_list = []
        criterions = []
        for j in range(args.num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss(reduction="none"))
            else:
                criterions.append(nn.KLDivLoss(reduction="none"))
            gate_name = best_trial.params.get(f"{i}_{j}_gate")
            if gate_name is None:
                raise RuntimeError(
                    f"Gate {i}_{j}_gate not found in trial {best_trial.number:04}"
                )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)

        # Same logic as dcl_train.py
        all_cutoff = all(g.__class__.__name__ == "CutoffGate" for g in gates_list)

        model_name = model_names[i]
        model = getattr(cifar_models, model_name)(num_classes).cuda()

        # Load pretrained checkpoint (when all inputs are Cutoff and i!=0)
        if all_cutoff and i != 0:
            pretrain_dir = os.path.join("checkpoint", "pre-train", model_name)
            checkpoint_path = os.path.join(pretrain_dir, "best_checkpoint.pkl")
            load_checkpoint(model=model, checkpoint_path=checkpoint_path)

        writer = SummaryWriter(
            f"runs/dcl_{args.num_nodes}/{best_trial.number:04}/{i}_{model_name}"
        )
        save_dir = (
            f"checkpoint/dcl_{args.num_nodes}/{best_trial.number:04}/{i}_{model_name}"
        )

        optimizer = getattr(torch.optim, optim_setting["name"])(
            model.parameters(), **optim_setting["args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
            optimizer, **scheduler_setting["args"]
        )
        links = build_links(criterions, gates_list)

        learner = Learner(
            model=model,
            writer=writer,
            scaler=torch.amp.GradScaler("cuda"),
            save_dir=save_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            links=links,
            loss_meter=AverageMeter(),
            score_meter=AverageMeter(),
        )
        learners.append(learner)

    trainer = DistillationTrainer(
        learners=learners,
        max_epoch=max_epoch,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device,
        trial=None,
    )

    best_score = trainer.train()
    print("-")
    print(f"Best trial = {best_trial.number:04}")
    print(f"Node 0 (primary) best top1 = {best_score:.2f}%")


if __name__ == "__main__":
    main()
