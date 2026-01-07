"""Learning rate schedulers for training.

Implementation based on Hugging Face Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
"""

import math
from functools import partial
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
) -> float:
    """Compute learning rate multiplier for cosine schedule with warmup.

    Args:
        current_step: Current training step.
        num_warmup_steps: Number of steps for the warmup phase.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 creates half-cosine).

    Returns:
        Learning rate multiplier (0.0 to 1.0).
    """
    # Linear warmup phase: lr scales from 0 to initial_lr
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # Cosine annealing phase: lr decays from initial_lr to 0
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    # Formula: 0.5 * (1 + cos(π * num_cycles * 2 * progress))
    # This creates a smooth cosine decay from 1.0 to 0.0
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    The learning rate schedule has two phases:
    1. Linear warmup: lr increases linearly from 0 to initial_lr over num_warmup_steps
    2. Cosine annealing: lr decreases following cosine curve from initial_lr to 0

    This is a common schedule used in transformer training and self-supervised learning.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: Number of steps for the warmup phase.
            Common practice: num_warmup_steps = int(num_training_steps * 0.1)
        num_training_steps: Total number of training steps.
            Calculated as: (dataset_size / batch_size) * num_epochs
        num_cycles: Number of cosine waves in the schedule.
            - 0.5 (default): Half-cosine, decays smoothly from max to 0
            - 1.0: Full cosine cycle, goes to 0 and back to max
            - >1.0: Multiple oscillations
        last_epoch: Index of the last epoch when resuming training.
            Use -1 (default) for new training.

    Returns:
        LambdaLR scheduler with the cosine warmup schedule.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> num_steps = len(train_loader) * num_epochs
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=int(num_steps * 0.1),
        ...     num_training_steps=num_steps
        ... )
        >>> for epoch in range(num_epochs):
        ...     for batch in train_loader:
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # Update lr every step, not epoch!

    Reference:
        Hugging Face Transformers implementation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    """
    lr_lambda: Callable[[int], float] = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
