import random

import torch

from fast_audiomentations.transforms._impl._reverse_triton import (
    apply_reverse as _apply_reverse_triton,
)


class Reverse:
    """Time-reverse every row in the batch with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,  # noqa: ARG002 - reverse cannot run in-place safely.
    ) -> torch.Tensor:
        """Flip every sample row left-to-right with probability ``p``."""
        if random.random() < self.p:
            return _apply_reverse_triton(samples)
        return samples
