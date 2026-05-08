import random

import torch

from fast_audiomentations.transforms._impl._pointwise_triton import (
    apply_pointwise as _apply_pointwise_triton,
)


class PolarityInversion:
    """Negate every sample in the batch with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,
    ) -> torch.Tensor:
        """Multiply every sample by -1 with probability ``p``."""
        if random.random() < self.p:
            return _apply_pointwise_triton(
                samples,
                None,
                0.0,
                0.0,
                has_gain=False,
                has_polarity=True,
                has_clip=False,
                inplace=inplace,
            )
        return samples
