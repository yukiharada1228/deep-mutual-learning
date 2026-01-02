import os
import time
from dataclasses import dataclass, field
from typing import Optional

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcl.gates import CutoffGate
from dcl.utils import AverageMeter, accuracy, save_checkpoint


class DistillationLink(nn.Module):
    def __init__(self, criterion: nn.Module, gate: nn.Module):
        super(DistillationLink, self).__init__()
        self.criterion = criterion
        self.gate = gate

    def forward(
        self,
        target_output,
        label,
        source_output,
        epoch: int,
        is_self_link: bool,
    ):
        # Compute all losses per-sample (reduction='none')
        # Gate functions will handle averaging
        if is_self_link:
            loss = self.criterion(target_output, label)
            # For self-links, no additional information is needed for gate function
            return self.gate(loss, epoch)
        else:
            # Convert to proper format when using PyTorch's KLDivLoss
            if isinstance(self.criterion, nn.KLDivLoss):
                # KLDivLoss expects log-probabilities and probabilities
                target_log_prob = F.log_softmax(target_output, dim=-1)
                source_prob = F.softmax(source_output.detach(), dim=-1)
                loss = self.criterion(target_log_prob, source_prob)
                # With reduction="none", output shape is (batch_size, num_classes)
                # Sum over class dimension to get per-sample loss (batch_size,)
                loss = loss.sum(dim=-1)
            else:
                loss = self.criterion(target_output, source_output)

            # Pass teacher logits and label for CorrectGate
            # These arguments are ignored by other gate functions
            return self.gate(
                loss,
                epoch,
                teacher_logits=source_output,
                label=label,
            )


def build_links(
    criterions: list[nn.Module], gates: list[nn.Module]
) -> list[DistillationLink]:
    """
    Build a list of DistillationLink instances with simple length validation and one-sided broadcast.

    - If either `criterions` or `gates` has length 1 while the other has length N>1,
      it will be broadcast to length N.
    - Otherwise, their lengths must match.
    """
    if len(criterions) == 1 and len(gates) > 1:
        criterions = criterions * len(gates)
    if len(gates) == 1 and len(criterions) > 1:
        gates = gates * len(criterions)
    if len(criterions) != len(gates):
        raise ValueError(
            f"criterions({len(criterions)}) and gates({len(gates)}) must match in length "
            "or one of them must be length 1 for broadcasting"
        )
    return [DistillationLink(c, g) for c, g in zip(criterions, gates)]


class TotalLoss(nn.Module):
    def __init__(self, links: list[DistillationLink]):
        super(TotalLoss, self).__init__()
        # Store all incoming links
        self.incoming_links = nn.ModuleList(links)

    def forward(self, model_id, outputs, labels, epoch):
        if model_id < 0 or model_id >= len(outputs):
            raise ValueError(f"Invalid model_id: {model_id}")
        losses = []
        target_output = outputs[model_id]
        label = labels[model_id]
        for i, link in enumerate(self.incoming_links):
            if i == model_id:
                losses.append(link(target_output, label, None, epoch, True))
            else:
                losses.append(link(target_output, None, outputs[i], epoch, False))
        loss = torch.stack(losses).sum()
        return loss


@dataclass
class Learner:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.amp.GradScaler
    optimizer: Optimizer
    links: list[DistillationLink]
    composite_loss: TotalLoss = field(init=False)
    loss_meter: AverageMeter
    score_meter: AverageMeter
    scheduler: LRScheduler = None
    best_score: float = 0.0
    eval: nn.Module = accuracy
    save_dir: Optional[str] = None

    def __post_init__(self):
        self.composite_loss = TotalLoss(links=self.links)


class DistillationTrainer:
    def __init__(
        self,
        learners: list[Learner],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        trial=None,
    ):
        self.learners = learners
        for learner in learners:
            if learner.save_dir:
                os.makedirs(learner.save_dir, exist_ok=True)
        self.max_epoch = max_epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.trial = trial
        self.data_length = len(self.train_dataloader)

    def train_on_batch(self, image, label, epoch, num_iter):
        image = image.cuda()
        label = label.cuda()

        outputs = []
        labels = []
        for learner in self.learners:
            # Check if all links have CutoffGate
            # If so, use eval mode (for pre-trained teacher models)
            all_cutoff = all(
                isinstance(link.gate, CutoffGate) for link in learner.links
            )
            if all_cutoff:
                learner.model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    y = learner.model(image)
            else:
                learner.model.train()
                with torch.amp.autocast("cuda"):
                    y = learner.model(image)
            outputs.append(y)
            labels.append(label)

        for model_id, learner in enumerate(self.learners):
            with torch.amp.autocast("cuda"):
                loss = learner.composite_loss(model_id, outputs, labels, epoch)
                if learner.model.training:
                    learner.scaler.scale(loss).backward()
                    learner.scaler.step(learner.optimizer)
                    learner.optimizer.zero_grad()
                    learner.scaler.update()
            [top1] = learner.eval(outputs[model_id], labels[model_id], topk=(1,))
            learner.score_meter.update(top1.item(), labels[model_id].size(0))
            learner.loss_meter.update(loss.item(), labels[model_id].size(0))

    def test_on_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()

        outputs = []
        labels = []
        for learner in self.learners:
            learner.model.eval()
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    y = learner.model(image)
            outputs.append(y)
            labels.append(label)

        for model_id, learner in enumerate(self.learners):
            [top1] = learner.eval(outputs[model_id], labels[model_id], topk=(1,))
            learner.score_meter.update(top1.item(), labels[model_id].size(0))

    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            print("epoch %d" % epoch)
            start_time = time.time()

            for idx, (image, label) in enumerate(self.train_dataloader):
                self.train_on_batch(
                    image=image, label=label, epoch=epoch - 1, num_iter=idx
                )
            for model_id, learner in enumerate(self.learners):
                train_lr = learner.optimizer.param_groups[0]["lr"]
                train_loss = learner.loss_meter.avg
                train_score = learner.score_meter.avg
                learner.writer.add_scalar("train_lr", train_lr, epoch)
                learner.writer.add_scalar("train_loss", train_loss, epoch)
                learner.writer.add_scalar("train_score", train_score, epoch)
                if learner.scheduler is not None:
                    learner.scheduler.step()
                print(
                    "model_id: {0:}   loss :train={1:.3f}   score :train={2:.3f}".format(
                        model_id, train_loss, train_score
                    )
                )
                learner.loss_meter.reset()
                learner.score_meter.reset()

            for image, label in self.test_dataloader:
                self.test_on_batch(image=image, label=label)
            for model_id, learner in enumerate(self.learners):
                test_score = learner.score_meter.avg
                learner.writer.add_scalar("test_score", test_score, epoch)
                print(
                    "model_id: {0:}   score :test={1:.3f}".format(model_id, test_score)
                )
                if learner.best_score <= learner.score_meter.avg:
                    if learner.save_dir:
                        save_checkpoint(
                            learner.model, learner.save_dir, epoch, is_best=True
                        )
                    learner.best_score = learner.score_meter.avg
                if model_id == 0 and self.trial is not None:
                    self.trial.report(test_score, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                learner.score_meter.reset()
            elapsed_time = time.time() - start_time
            print("  elapsed_time:{0:.3f}[sec]".format(elapsed_time))

        for learner in self.learners:
            learner.writer.close()

        best_score = self.learners[0].best_score
        return best_score

    def __len__(self):
        return len(self.learners)
