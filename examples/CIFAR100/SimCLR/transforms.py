from typing import List

import torch
from PIL import Image
from torchvision import transforms


class SimCLRTransforms:
    """SimCLR data augmentation transforms (Appendix A).

    Applies a sequence of augmentations as specified in the SimCLR paper:
    - RandomResizedCrop: scale=(0.08, 1.0), ratio=(3/4, 4/3)
    - RandomHorizontalFlip: p=0.5
    - ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s) with p=0.8
    - RandomGrayscale: p=0.2
    - GaussianBlur: p=0.5, sigma=[0.1, 2.0], kernel_size=10% of height/width (odd, min 3)
    - ToTensor

    Args:
        input_size: Input image size (height/width for square images).
        s: Color distortion strength parameter.
        include_blur: Whether to include Gaussian blur augmentation.
    """

    # Default augmentation probabilities
    HORIZONTAL_FLIP_PROB = 0.5
    COLOR_JITTER_PROB = 0.8
    GRAYSCALE_PROB = 0.2
    GAUSSIAN_BLUR_PROB = 0.5

    # Color jitter multipliers
    BRIGHTNESS_FACTOR = 0.8
    CONTRAST_FACTOR = 0.8
    SATURATION_FACTOR = 0.8
    HUE_FACTOR = 0.2

    # Gaussian blur parameters
    BLUR_KERNEL_SIZE_RATIO = 0.1
    MIN_KERNEL_SIZE = 3
    BLUR_SIGMA_RANGE = (0.1, 2.0)

    def __init__(
        self, input_size: int = 32, s: float = 0.5, include_blur: bool = False
    ) -> None:
        self.input_size = input_size
        self.s = s
        self.include_blur = include_blur
        self.train_transform = self._build_transform()

    def _calculate_blur_kernel_size(self) -> int:
        """Calculate odd kernel size for Gaussian blur based on input size."""
        kernel_size = max(
            self.MIN_KERNEL_SIZE,
            int(round(self.BLUR_KERNEL_SIZE_RATIO * self.input_size)),
        )
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _build_color_jitter(self) -> transforms.ColorJitter:
        """Build color jitter transform with specified strength."""
        return transforms.ColorJitter(
            brightness=self.BRIGHTNESS_FACTOR * self.s,
            contrast=self.CONTRAST_FACTOR * self.s,
            saturation=self.SATURATION_FACTOR * self.s,
            hue=self.HUE_FACTOR * self.s,
        )

    def _build_transform(self) -> transforms.Compose:
        """Build the complete SimCLR augmentation pipeline."""
        augmentations = [
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(p=self.HORIZONTAL_FLIP_PROB),
            transforms.RandomApply(
                [self._build_color_jitter()], p=self.COLOR_JITTER_PROB
            ),
            transforms.RandomGrayscale(p=self.GRAYSCALE_PROB),
        ]

        if self.include_blur:
            kernel_size = self._calculate_blur_kernel_size()
            augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=kernel_size, sigma=self.BLUR_SIGMA_RANGE
                        )
                    ],
                    p=self.GAUSSIAN_BLUR_PROB,
                )
            )

        augmentations.append(transforms.ToTensor())
        return transforms.Compose(augmentations)

    def __call__(self, x: Image.Image) -> List[torch.Tensor]:
        """Apply two independent random augmentations to create query and key views.

        Args:
            x: Input PIL Image.

        Returns:
            List containing two augmented views [query, key].
        """
        q = self.train_transform(x)
        k = self.train_transform(x)
        return [q, k]
