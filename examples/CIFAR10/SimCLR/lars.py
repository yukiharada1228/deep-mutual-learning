"""LARS (Layer-wise Adaptive Rate Scaling) optimizer for large-batch SSL training.

Reference: https://arxiv.org/abs/1708.03888
"""

from typing import Iterable, Optional

import torch
from torch import optim


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient (default: 0).
        momentum: Momentum factor (default: 0.9).
        eta: LARS trust coefficient (default: 0.001).
        weight_decay_filter: Exclude bias/norm from weight decay (default: False).
        lars_adaptation_filter: Exclude bias/norm from LARS adaptation (default: False).
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
        return param.ndim == 1

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                if group["weight_decay"] != 0:
                    if not group["weight_decay_filter"] or not self._is_bias_or_norm(
                        param
                    ):
                        grad = grad.add(param, alpha=group["weight_decay"])

                if not group["lars_adaptation_filter"] or not self._is_bias_or_norm(
                    param
                ):
                    param_norm = torch.norm(param)
                    grad_norm = torch.norm(grad)
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

                param_state = self.state[param]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(param)

                buf = param_state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                param.add_(buf, alpha=-group["lr"])

        return loss
