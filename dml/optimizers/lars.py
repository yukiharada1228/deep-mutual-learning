"""LARS (Layer-wise Adaptive Rate Scaling) Optimizer.

Implementation based on:
- Paper: "Large Batch Training of Convolutional Networks" (https://arxiv.org/abs/1708.03888)
- Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L224
"""

from typing import Iterable, Optional

import torch
from torch import optim


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    LARS is an extension of SGD with momentum that computes a separate learning rate
    per layer by normalizing gradients by L2 norm and scaling by weight L2 norm.
    This decouples the magnitude of updates from gradient magnitude, enabling
    stable large-batch training.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (required).
        weight_decay: Weight decay coefficient (default: 0).
        momentum: Momentum factor (default: 0.9).
        eta: LARS trust coefficient for computing adaptive lr (default: 0.001).
        weight_decay_filter: If True, exclude bias and norm layers from weight decay (default: False).
        lars_adaptation_filter: If True, exclude bias and norm layers from LARS adaptation (default: False).

    Reference:
        You, Yang, Igor Gitman, and Boris Ginsburg.
        "Large batch training of convolutional networks."
        arXiv preprint arXiv:1708.03888 (2017).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: bool = False,
        lars_adaptation_filter: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta value: {eta}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _is_bias_or_norm(param: torch.Tensor) -> bool:
        """Check if parameter is a bias or batch norm parameter.

        Bias and batch norm parameters are 1D tensors.

        Args:
            param: Parameter tensor to check.

        Returns:
            True if parameter is 1D (bias or batch norm), False otherwise.
        """
        return param.ndim == 1

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply weight decay (L2 regularization)
                # Skip for bias/norm parameters if weight_decay_filter is enabled
                if group["weight_decay"] != 0:
                    if not group["weight_decay_filter"] or not self._is_bias_or_norm(
                        param
                    ):
                        grad = grad.add(param, alpha=group["weight_decay"])

                # Compute LARS adaptive learning rate
                # Skip for bias/norm parameters if lars_adaptation_filter is enabled
                if not group["lars_adaptation_filter"] or not self._is_bias_or_norm(
                    param
                ):
                    param_norm = torch.norm(param)
                    grad_norm = torch.norm(grad)

                    # Compute trust ratio: eta * ||w|| / ||∇w||
                    # Use 1.0 if either norm is zero to avoid division by zero
                    trust_ratio = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            grad_norm > 0.0,
                            group["eta"] * param_norm / grad_norm,
                            torch.ones_like(param_norm),
                        ),
                        torch.ones_like(param_norm),
                    )
                    grad = grad.mul(trust_ratio)

                # Apply momentum
                param_state = self.state[param]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(param)

                momentum_buffer = param_state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(grad)

                # Update parameters
                param.add_(momentum_buffer, alpha=-group["lr"])

        return loss
