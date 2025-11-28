# Import packages
import argparse
import os

import optuna
import torch
import torch.nn as nn
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.study import MaxTrialsCallback
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg import KnowledgeTransferGraph, Node, build_edges, gates
from ktg.dataset.cifar_datasets.cifar100 import get_datasets
from ktg.losses import KLDivLoss
from ktg.models import cifar_models
from ktg.utils import AverageMeter, WorkerInitializer, load_checkpoint, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=6)
parser.add_argument("--n_trials", default=1500)
parser.add_argument(
    "--models",
    default=["resnet32", "resnet110", "wideresnet28_2"],
    nargs="*",
    type=str,
)
parser.add_argument(
    "--gates",
    default=["ThroughGate", "CutoffGate", "PositiveLinearGate", "NegativeLinearGate"],
    nargs="*",
    type=str,
)

args = parser.parse_args()
manualSeed = int(args.seed)
num_nodes = int(args.num_nodes)
n_trials = int(args.n_trials)
models_name = args.models
gates_name = args.gates


def objective(trial):
    # Fix the seed value
    set_seed(manualSeed)

    # Prepare the CIFAR-100 for training
    batch_size = 64
    num_workers = 10

    train_dataset, val_dataset = get_datasets()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
    )

    # Prepare for training
    max_epoch = 200

    optim_setting = {
        "name": "SGD",
        "args": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
        },
    }
    scheduler_setting = {
        "name": "CosineAnnealingLR",
        "args": {"T_max": max_epoch, "eta_min": 0.0},
    }

    num_classes = 100
    nodes = []
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss())
            else:
                criterions.append(KLDivLoss())
            gate_name = trial.suggest_categorical(
                f"{i}_{j}_gate",
                gates_name,
            )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)
        if i == 0:
            model_name = models_name[0]
        else:
            model_name = trial.suggest_categorical(f"{i}_model", models_name)
        model = getattr(cifar_models, model_name)(num_classes).cuda()
        if (
            all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list)
            and i != 0
        ):
            load_checkpoint(
                model=model, save_dir=f"checkpoint/pre-train/{model_name}", is_best=True
            )
        writer = SummaryWriter(
            f"runs/dcl_{num_nodes}/{trial.number:04}/{i}_{model_name}"
        )
        optimizer = getattr(torch.optim, optim_setting["name"])(
            model.parameters(), **optim_setting["args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
            optimizer, **scheduler_setting["args"]
        )
        edges = build_edges(criterions, gates_list)

        node = Node(
            model=model,
            writer=writer,
            scaler=torch.amp.GradScaler("cuda"),
            optimizer=optimizer,
            scheduler=scheduler,
            edges=edges,
            loss_meter=AverageMeter(),
            score_meter=AverageMeter(),
        )
        nodes.append(node)

    graph = KnowledgeTransferGraph(
        nodes=nodes,
        max_epoch=max_epoch,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        trial=trial,
    )
    best_score = graph.train()
    return best_score


if __name__ == "__main__":
    # Cteate study object
    study_name = f"dcl_{num_nodes}"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    # Start optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[
            MaxTrialsCallback(
                n_trials,
                states=None,
            )
        ],
    )
