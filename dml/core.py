import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLink(nn.Module):
    """
    A link for knowledge distillation with temperature scaling.

    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss, nn.KLDivLoss)
        temperature: Temperature for knowledge distillation (default: 1.0)
                    When T > 1, softened probabilities are used for KL divergence
    """

    def __init__(
        self,
        criterion: nn.Module,
        temperature: float | None = 1.0,
    ):
        super(DistillationLink, self).__init__()
        self.criterion = criterion
        self.temperature = temperature

    def forward(
        self,
        target_output,
        label,
        source_output=None,
    ):
        """
        Compute distillation loss.

        Args:
            target_output: Output logits from the target model
            label: Ground truth labels (used for self-link)
            source_output: Output logits from the source model (None for self-link)

        Returns:
            Loss value (gate is checked via is_active() before calling this)
        """
        # Self-link: supervised learning
        if source_output is None:
            return self.criterion(target_output, label)

        # Cross-link: knowledge distillation
        if isinstance(self.criterion, nn.KLDivLoss):
            # Temperature-scaled KL divergence
            # Following Hinton et al.: gradient ∂C/∂zi ≈ 1/(NT²)(zi - vi)
            target_log_prob = F.log_softmax(target_output / self.temperature, dim=-1)
            source_prob = F.softmax(source_output.detach() / self.temperature, dim=-1)
            loss = self.criterion(target_log_prob, source_prob)
            # Scale by T² to maintain gradient magnitude
            return loss * (self.temperature**2)
        else:
            return self.criterion(target_output, source_output)


class CompositeLoss(nn.Module):
    """
    Composite loss for multi-node knowledge distillation.

    Combines supervised loss (self-link) and distillation losses (cross-links).
    """

    def __init__(self, model_id: int, links: list[DistillationLink]):
        super(CompositeLoss, self).__init__()
        self.model_id = model_id
        self.incoming_links = nn.ModuleList(links)

    def forward(self, outputs, label):
        """
        Compute composite loss for the bound model.

        Args:
            outputs: List of output logits from all models
            label: Ground truth labels (shared across all nodes)

        Returns:
            Combined loss (supervised + distillation)
        """
        target_output = outputs[self.model_id]

        # Supervised Loss (Self-link)
        supervised_loss = self.incoming_links[self.model_id](target_output, label, None)

        # Distillation Loss (Cross-links)
        distillation_losses = [
            link(target_output, None, outputs[i])
            for i, link in enumerate(self.incoming_links)
            if i != self.model_id
        ]

        if distillation_losses:
            distillation_loss_mean = torch.stack(distillation_losses).mean()
        else:
            distillation_loss_mean = torch.zeros_like(supervised_loss)

        return supervised_loss + distillation_loss_mean


def build_links(
    criterions: list[nn.Module],
    temperatures: list[float] | None = None,
) -> list[DistillationLink]:
    """
    Build a list of DistillationLink instances.

    Args:
        criterions: List of loss functions
        temperatures: List of temperatures (default: all 1.0)

    Returns:
        List of DistillationLink instances
    """
    n = len(criterions)

    if temperatures is None:
        temperatures = [1.0] * n

    return [DistillationLink(c, t) for c, t in zip(criterions, temperatures)]


def build_mutual_learning_losses(
    num_nodes: int,
    temperature: float = 1.0,
    supervised_factory=None,
    distillation_factory=None,
) -> list[CompositeLoss]:
    """
    Build one CompositeLoss per node for symmetric deep mutual learning.

    Args:
        num_nodes: Number of mutually learning nodes.
        temperature: Distillation temperature for cross-links.
        supervised_factory: Optional callable returning the self-link criterion.
        distillation_factory: Optional callable returning the cross-link criterion.

    Returns:
        List of CompositeLoss instances, one per target node.
    """
    if supervised_factory is None:
        supervised_factory = lambda: nn.CrossEntropyLoss(reduction="mean")
    if distillation_factory is None:
        distillation_factory = lambda: nn.KLDivLoss(reduction="batchmean")

    losses = []
    for target_index in range(num_nodes):
        criterions = []
        temperatures = []
        for source_index in range(num_nodes):
            if target_index == source_index:
                criterions.append(supervised_factory())
                temperatures.append(None)
            else:
                criterions.append(distillation_factory())
                temperatures.append(temperature)
        losses.append(CompositeLoss(target_index, build_links(criterions, temperatures)))
    return losses
