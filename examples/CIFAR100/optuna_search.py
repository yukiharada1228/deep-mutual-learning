import argparse
import logging
import os

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage
import torch.nn as nn
from dml import Callback, Edge, EpochState, Graph, Node, Trainer
from dml.utils import accuracy, set_seed

from training_utils import (CIFAR100_NUM_CLASSES, create_cifar100_dataloaders,
                            create_grad_scaler, create_model, create_optimizer,
                            create_scheduler, get_device)


def build_graph(trial, args, device):
    num_nodes = args.num_nodes
    trial.set_user_attr("num_nodes", num_nodes)

    model_names = [args.model] + [
        trial.suggest_categorical(f"model_{i}", args.models)
        for i in range(1, num_nodes)
    ]

    edges = []
    for i in range(num_nodes):
        if trial.suggest_categorical(f"label_edge_{i}", [True, False]):
            edges.append(Edge(None, i, nn.CrossEntropyLoss()))

    for src in range(num_nodes):
        for tgt in range(num_nodes):
            if src == tgt:
                continue
            if trial.suggest_categorical(f"peer_{src}_{tgt}", [True, False]):
                edges.append(
                    Edge(
                        src, tgt, nn.KLDivLoss(reduction="batchmean"), args.temperature
                    )
                )

    nodes = []
    for model_name in model_names:
        model = create_model(model_name, device, CIFAR100_NUM_CLASSES)
        optimizer = create_optimizer(model, lr=args.lr, wd=args.wd)
        ckpt_path = os.path.join(
            "checkpoint", "independent", model_name, "latest_checkpoint.pth"
        )
        nodes.append(
            Node(
                model=model,
                optimizer=optimizer,
                scheduler=create_scheduler(optimizer, max_epoch=args.epochs),
                scaler=create_grad_scaler(device),
                checkpoint_path=ckpt_path,
            )
        )

    if not any(e.target == 0 for e in edges):
        raise optuna.TrialPruned()

    return Graph(nodes, edges)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna search for model/loss configurations on CIFAR-100"
    )
    parser.add_argument("--model", default="resnet32", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument(
        "--models",
        default=["resnet32", "resnet110", "wideresnet28_2"],
        nargs="+",
        type=str,
        help="Model pool for extra nodes",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int, help="Epochs per trial")
    parser.add_argument(
        "--trials", default=1500, type=int, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--num-nodes",
        default=7,
        type=int,
        help="Total number of nodes including target node (default: 2)",
    )
    parser.add_argument("--study-name", default="dml_search", type=str)
    parser.add_argument("--storage", default="optuna_journal.log", type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader = create_cifar100_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    def objective(trial):
        graph = build_graph(trial, args, device)
        num_nodes = len(graph)
        best = 0.0
        best_scores = [0.0] * num_nodes

        class PruningCallback(Callback):
            def on_epoch_end(self, _, state: EpochState):
                nonlocal best, best_scores
                score = state.val_scores[0]
                best = max(best, score)
                for i, s in enumerate(state.val_scores):
                    best_scores[i] = max(best_scores[i], s)
                trial.report(score, state.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        Trainer(
            graph=graph,
            device=device,
            score_fn=lambda output, target: accuracy(output, target, topk=(1,))[
                0
            ].item(),
            callbacks=[PruningCallback()],
        ).fit(train_loader, val_loader, epochs=args.epochs)

        for i, s in enumerate(best_scores):
            trial.set_user_attr(f"best_acc_{i}", s)

        return best

    study = optuna.create_study(
        study_name=args.study_name,
        storage=JournalStorage(JournalFileBackend(args.storage)),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.epochs),
        load_if_exists=True,
    )
    study.set_user_attr("main_model", args.model)
    study.optimize(objective, n_trials=args.trials)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val acc (node 0): {study.best_value:.2f}%")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
