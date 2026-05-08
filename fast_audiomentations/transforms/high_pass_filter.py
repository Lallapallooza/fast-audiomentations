import random

import torch

from fast_audiomentations.transforms._impl._filter_triton import (
    create_filters as _create_high_pass_filters,
)
from fast_audiomentations.transforms._impl._filter_triton import (
    fft_conv1d as _fft_conv1d,
)


class HighPassFilter:
    """Random-cutoff high-pass FIR filter applied via FFT-based conv1d."""

    def __init__(
        self,
        min_cutoff_freq: int = 500,
        max_cutoff_freq: int = 2000,
        num_taps: int = 101,
        buffer_size: int = 129,
        p: float = 0.5,
    ) -> None:
        self.__min_cutoff_freq = min_cutoff_freq
        self.__max_cutoff_freq = max_cutoff_freq
        self.p = p
        self.num_taps = num_taps
        self.window = torch.hamming_window(
            num_taps, device="cuda", dtype=torch.float32, periodic=False
        )
        self.random_buffer = torch.empty(buffer_size, device="cuda")
        half = (num_taps - 1) // 2
        self.time = torch.arange(
            -half, half + 1, dtype=torch.float32, device="cuda"
        )
        self.filter_output = torch.empty(
            (buffer_size, self.num_taps), device="cuda", dtype=torch.float32
        )

    def __generate_random_cutoffs(self, num_audios: int) -> torch.Tensor:
        assert num_audios <= self.random_buffer.size(0)

        buff_slice = self.random_buffer[:num_audios]
        buff_slice.uniform_(self.__min_cutoff_freq, self.__max_cutoff_freq)

        return buff_slice

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        inplace: bool = False,  # noqa: ARG002 - filter path is FFT-shaped, no inplace.
    ) -> torch.Tensor:
        """Convolve ``samples`` with a random-cutoff high-pass FIR filter with probability ``p``."""
        if random.random() < self.p:
            freqs = self.__generate_random_cutoffs(samples.shape[0])

            buff_slice = self.filter_output[: len(freqs)]
            _create_high_pass_filters(
                buff_slice,
                freqs,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                "high",
            )
            return _fft_conv1d(samples, buff_slice)

        return samples
