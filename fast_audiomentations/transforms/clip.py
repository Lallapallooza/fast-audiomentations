import random

import torch

from fast_audiomentations.transforms._impl._clip_triton import (
    apply_clip as _apply_clip_triton,
)


class Clip:
    """Hard-clip every sample in the batch into a fixed [min, max] range."""

    def __init__(
        self,
        min: float = -1.0,  # noqa: A002 - matches torch.clip / numpy.clip kwarg name.
        max: float = 1.0,  # noqa: A002
        p: float = 0.5,
    ) -> None:
        self.__min = min
        self.__max = max
        self.p = p

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,
    ) -> torch.Tensor:
        """Hard-clip every row of ``samples`` into ``[min, max]`` with probability ``p``."""
        if random.random() < self.p:
            return _apply_clip_triton(
                samples, self.__min, self.__max, inplace=inplace
            )
        return samples
