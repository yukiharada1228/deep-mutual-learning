import logging
import time
from typing import Callable

import torch

from .callbacks import Callback, EpochState
from .graph import Graph
from .utils import AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
    """Unified trainer for any task.

    Evaluation is per-node: each :class:`~dml.Node` may carry its own
    ``eval_fn`` (a callable ``(model, device) → float``).  This enables
    heterogeneous graphs where different nodes use different evaluation
    strategies (e.g. classification accuracy vs. KNN).

    Args:
        graph:         The learning graph.
        device:        Torch device to use.
        prepare_batch: Transforms a raw DataLoader batch into a ``dict``
                       consumed by ``Node.forward`` and ``Edge.compute``.
                       Default converts ``(image, label)`` →
                       ``{"image": …, "label": …}``.
        callbacks:     List of ``Callback`` instances.
    """

    def __init__(
        self,
        graph: Graph,
        device: torch.device,
        prepare_batch: Callable | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.graph = graph
        self.device = device
        self.prepare_batch = prepare_batch or self._default_prepare_batch
        self.callbacks = callbacks or []

    def _default_prepare_batch(self, raw_batch) -> dict:
        image, label = raw_batch
        return {"image": image.to(self.device), "label": label.to(self.device)}

    def fit(self, train_dataloader, epochs: int = 1) -> None:
        device_type = self.device.type

        for callback in self.callbacks:
            callback.on_train_begin(self.graph)

        logger.info("Training started")
        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d", epoch, epochs)
            start_time = time.time()

            train_loss_meters = [AverageMeter() for _ in self.graph]

            self.graph.train_all()
            for raw_batch in train_dataloader:
                batch = self.prepare_batch(raw_batch)
                outputs = self.graph.forward_all(batch, device_type)
                losses = self.graph.compute_losses(outputs, batch)
                self.graph.optimize_all(losses)
                self.graph.step_schedulers("step")

                batch_size = self._get_batch_size(batch)
                for model_id, loss in enumerate(losses):
                    if loss is not None:
                        train_loss_meters[model_id].update(loss.item(), batch_size)

            self.graph.step_schedulers("epoch")

            train_losses = [m.avg for m in train_loss_meters]
            learning_rates = [node.lr for node in self.graph]

            for model_id in range(len(self.graph)):
                if not self.graph.is_teacher(model_id):
                    logger.info(
                        "  Model %d: loss=%.4f, lr=%.6f",
                        model_id,
                        train_losses[model_id],
                        learning_rates[model_id],
                    )

            # Per-node evaluation
            val_scores = [0.0] * len(self.graph)
            for model_id, node in enumerate(self.graph):
                if self.graph.is_teacher(model_id) or node.eval_fn is None:
                    continue
                score = node.eval_fn(node.model, self.device)
                val_scores[model_id] = score
                logger.info("  Model %d: val_score=%.2f%%", model_id, score)

            state = EpochState(
                epoch=epoch,
                train_losses=train_losses,
                val_scores=val_scores,
                learning_rates=learning_rates,
                elapsed_time=time.time() - start_time,
            )

            for callback in self.callbacks:
                callback.on_epoch_end(self.graph, state)

            logger.info("  Elapsed time: %.2fs", state.elapsed_time)

        for callback in self.callbacks:
            callback.on_train_end(self.graph)

        logger.info("Training completed")

    @staticmethod
    def _get_batch_size(batch: dict) -> int:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                return v[0].size(0)
        return 1
