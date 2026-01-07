"""Learning rate schedulers for deep mutual learning and self-supervised learning."""

from .cosine_warmup import get_cosine_schedule_with_warmup

__all__ = ["get_cosine_schedule_with_warmup"]
