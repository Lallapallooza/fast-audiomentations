import random

import torch

from fast_audiomentations.transforms._impl._pointwise_triton import (
    apply_pointwise as _apply_pointwise_triton,
)


class Gain:
    """Multiply every sample by a random per-row gain in dB."""

    def __init__(
        self,
        min_gain_in_db: float = -12,
        max_gain_in_db: float = 12,
        p: float = 0.5,
        buffer_size: int = 129,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        self.p = p
        self.dtype = dtype

        self.random_buffer = torch.empty(
            buffer_size, device="cuda", dtype=dtype
        )

    def _generate_random_amplitude_ratios(
        self, num_audios: int
    ) -> torch.Tensor:
        # Sample dB in-place, convert to amplitude ratio (10^(dB/20)
        # = exp(dB * ln(10)/20)) so the kernel multiplies directly.
        assert num_audios <= self.random_buffer.size(0)

        buf_slice = self.random_buffer[:num_audios]
        buf_slice.uniform_(self.min_gain_in_db, self.max_gain_in_db)
        buf_slice.mul_(0.11512925464970228).exp_()

        return buf_slice

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,
    ) -> torch.Tensor:
        """Multiply ``samples`` by a random per-row gain in dB with probability ``p``."""
        if random.random() < self.p:
            gain_factors = self._generate_random_amplitude_ratios(
                samples.shape[0]
            )
            return _apply_pointwise_triton(
                samples,
                gain_factors,
                0.0,
                0.0,
                has_gain=True,
                has_polarity=False,
                has_clip=False,
                inplace=inplace,
            )
        return samples
