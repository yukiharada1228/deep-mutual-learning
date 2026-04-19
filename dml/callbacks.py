from dataclasses import dataclass
from typing import Any

from .utils import save_checkpoint


@dataclass
class EpochState:
    epoch: int
    train_losses: list[float]
    train_scores: list[float]
    val_scores: list[float]
    learning_rates: list[float]
    elapsed_time: float


class DMLCallback:
    def on_train_begin(self, session):
        pass

    def on_epoch_end(self, session, state: EpochState):
        pass

    def on_train_end(self, session):
        pass


class TensorBoardCallback(DMLCallback):
    def __init__(self, writers: list[Any]):
        self.writers = writers

    def on_epoch_end(self, session, state: EpochState):
        for model_id, writer in enumerate(self.writers):
            writer.add_scalar("train_lr", state.learning_rates[model_id], state.epoch)
            writer.add_scalar("train_loss", state.train_losses[model_id], state.epoch)
            writer.add_scalar("train_score", state.train_scores[model_id], state.epoch)
            writer.add_scalar("test_score", state.val_scores[model_id], state.epoch)

    def on_train_end(self, session):
        for writer in self.writers:
            writer.close()


class CheckpointCallback(DMLCallback):
    def __init__(self, save_dirs: list[str]):
        self.save_dirs = save_dirs

    def on_epoch_end(self, session, state: EpochState):
        for model_id, node in enumerate(session):
            save_checkpoint(
                node.model,
                self.save_dirs[model_id],
                state.epoch,
                filename="latest_checkpoint.pth",
            )
