import logging
import time
from typing import Callable

import torch

from .callbacks import DMLCallback, EpochState
from .runtime import DMLSession
from .utils import AverageMeter

logger = logging.getLogger(__name__)


class DMLTrainer:
    def __init__(
        self,
        session: DMLSession,
        device: torch.device,
        score_fn: Callable[[torch.Tensor, torch.Tensor], float],
        callbacks: list[DMLCallback] | None = None,
    ):
        self.session = session
        self.device = device
        self.score_fn = score_fn
        self.callbacks = callbacks or []

    def fit(self, train_dataloader, val_dataloader, epochs: int) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(self.session)

        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d", epoch, epochs)
            start_time = time.time()

            train_loss_meters = [AverageMeter() for _ in self.session]
            train_score_meters = [AverageMeter() for _ in self.session]

            for image, label in train_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)

                for node in self.session:
                    node.train()

                outputs = self.session.forward_all(image, self.device.type)
                losses = self.session.compute_losses(outputs, label)
                self.session.optimize_all(losses)

                for model_id, loss in enumerate(losses):
                    score = self.score_fn(outputs[model_id], label)
                    train_score_meters[model_id].update(score, label.size(0))
                    train_loss_meters[model_id].update(loss.item(), label.size(0))

            train_losses = [meter.avg for meter in train_loss_meters]
            train_scores = [meter.avg for meter in train_score_meters]
            learning_rates = [node.lr for node in self.session]

            for model_id, train_loss in enumerate(train_losses):
                logger.info(
                    "  Model %d: loss=%.4f, acc=%.2f%%",
                    model_id, train_loss, train_scores[model_id],
                )

            self.session.step_schedulers()

            val_score_meters = [AverageMeter() for _ in self.session]
            for node in self.session:
                node.eval()
            for image, label in val_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)

                for model_id, node in enumerate(self.session):
                    with torch.no_grad():
                        output = node.forward(image, self.device.type)

                    score = self.score_fn(output, label)
                    val_score_meters[model_id].update(score, label.size(0))

            state = EpochState(
                epoch=epoch,
                train_losses=train_losses,
                train_scores=train_scores,
                val_scores=[meter.avg for meter in val_score_meters],
                learning_rates=learning_rates,
                elapsed_time=time.time() - start_time,
            )

            for callback in self.callbacks:
                callback.on_epoch_end(self.session, state)

            for model_id, val_score in enumerate(state.val_scores):
                logger.info("  Model %d: test_acc=%.2f%%", model_id, val_score)

            logger.info("  Elapsed time: %.2fs", state.elapsed_time)

        for callback in self.callbacks:
            callback.on_train_end(self.session)
