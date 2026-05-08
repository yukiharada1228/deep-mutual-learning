"""Device and AMP utilities."""

import torch


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Create a :class:`GradScaler` that is only enabled on CUDA."""
    return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
