"""Compose Clip / Gain / PolarityInversion into a single fused Triton launch."""

from __future__ import annotations

import random

import torch

from fast_audiomentations.transforms._impl._pointwise_triton import (
    apply_pointwise as _apply_pointwise_triton,
)
from fast_audiomentations.transforms.clip import Clip
from fast_audiomentations.transforms.gain import Gain
from fast_audiomentations.transforms.polarity_inversion import (
    PolarityInversion,
)


class FusedPointwise:
    """Apply Clip / Gain / PolarityInversion in one Triton launch.

    Components are independent: each may be ``None`` (disabled) or carry its
    own probability ``p``. At call time, the active set is resolved per
    component and dispatched to one of up to eight specialised kernel
    binaries (``HAS_GAIN`` x ``HAS_POLARITY`` x ``HAS_CLIP`` constexpr
    flags). The kernel applies enabled ops in the fixed order
    ``gain -> polarity -> clip``; that matches the typical audio chain
    (scale, then flip, then clamp). When all coins miss, the input is
    returned unchanged with no kernel launch.

    @param gain: a :class:`Gain` instance, or ``None``.
    @param polarity: a :class:`PolarityInversion` instance, or ``None``.
    @param clip: a :class:`Clip` instance, or ``None``.
    """

    def __init__(
        self,
        *,
        gain: Gain | None = None,
        polarity: PolarityInversion | None = None,
        clip: Clip | None = None,
    ) -> None:
        if gain is None and polarity is None and clip is None:
            raise ValueError("FusedPointwise requires at least one component.")
        self.gain = gain
        self.polarity = polarity
        self.clip = clip

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - cross-transform API uniformity.
        inplace: bool = False,
    ) -> torch.Tensor:
        """Apply the fused chain to ``samples`` with per-component ``p``."""
        has_gain = self.gain is not None and random.random() < self.gain.p
        has_polarity = (
            self.polarity is not None and random.random() < self.polarity.p
        )
        has_clip = self.clip is not None and random.random() < self.clip.p

        if not (has_gain or has_polarity or has_clip):
            return samples

        gain_factors: torch.Tensor | None = None
        if has_gain:
            assert self.gain is not None
            gain_factors = self.gain._generate_random_amplitude_ratios(
                samples.shape[0]
            )

        if has_clip:
            assert self.clip is not None
            clip_min = self.clip.min
            clip_max = self.clip.max
        else:
            clip_min = 0.0
            clip_max = 0.0

        return _apply_pointwise_triton(
            samples,
            gain_factors,
            clip_min,
            clip_max,
            has_gain=has_gain,
            has_polarity=has_polarity,
            has_clip=has_clip,
            inplace=inplace,
        )
