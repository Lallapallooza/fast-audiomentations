import random

import torch

from fast_audiomentations.transforms._impl._pointwise_triton import (
    apply_pointwise as _apply_pointwise_triton,
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

    @property
    def min(self) -> float:
        """Lower clip bound."""
        return self.__min

    @property
    def max(self) -> float:
        """Upper clip bound."""
        return self.__max

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,
    ) -> torch.Tensor:
        """Hard-clip every row of ``samples`` into ``[min, max]`` with probability ``p``."""
        if random.random() < self.p:
            return _apply_pointwise_triton(
                samples,
                None,
                self.__min,
                self.__max,
                has_gain=False,
                has_polarity=False,
                has_clip=True,
                inplace=inplace,
            )
        return samples
