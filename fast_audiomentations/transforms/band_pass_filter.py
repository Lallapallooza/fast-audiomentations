import random

import torch

from fast_audiomentations.transforms._impl._filter_triton import (
    create_filters as _create_pass_filters,
)
from fast_audiomentations.transforms._impl._filter_triton import (
    fft_conv1d as _fft_conv1d,
)


class BandPassFilter:
    """Random-bandwidth band-pass FIR filter applied via FFT-based conv1d."""

    def __init__(
        self,
        min_center_freq: int = 500,
        max_center_freq: int = 2000,
        num_taps: int = 101,
        buffer_size: int = 129,
        p: float = 0.5,
    ) -> None:
        self.__min_center_freq = min_center_freq
        self.__max_center_freq = max_center_freq
        self.p = p
        self.num_taps = num_taps

        self.window = torch.hamming_window(
            num_taps, device="cuda", dtype=torch.float32, periodic=False
        )

        self.random_buffer_low = torch.empty(buffer_size, device="cuda")
        self.random_buffer_high = torch.empty(buffer_size, device="cuda")

        half = (num_taps - 1) // 2
        self.time = torch.arange(
            -half, half + 1, dtype=torch.float32, device="cuda"
        )

        self.filter_output_low = torch.empty(
            (buffer_size, self.num_taps), device="cuda", dtype=torch.float32
        )
        self.filter_output_high = torch.empty(
            (buffer_size, self.num_taps), device="cuda", dtype=torch.float32
        )

    def __generate_random_cutoffs(
        self, num_audios: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert num_audios <= self.random_buffer_low.size(0)

        buff_slice_low = self.random_buffer_low[:num_audios]
        buff_slice_low.uniform_(self.__min_center_freq, self.__max_center_freq)

        buff_slice_high = self.random_buffer_high[:num_audios]
        buff_slice_high.uniform_(
            self.__min_center_freq, self.__max_center_freq
        )

        # TODO: rewrite on triton as well
        # High cutoff is the low cutoff plus a positive bandwidth, capped.
        buff_slice_high = torch.clip(
            buff_slice_high + buff_slice_low, 0, self.__max_center_freq
        )

        return buff_slice_low, buff_slice_high

    def __call__(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        inplace: bool = False,  # noqa: ARG002 - filter path is FFT-shaped, no inplace.
    ) -> torch.Tensor:
        """Convolve ``samples`` with a random-bandwidth band-pass FIR filter with probability ``p``."""
        if random.random() < self.p:
            freqs_low, freqs_high = self.__generate_random_cutoffs(
                samples.shape[0]
            )

            buff_slice_low = self.filter_output_low[: len(freqs_low)]
            _create_pass_filters(
                buff_slice_low,
                freqs_low,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                mode="low",
            )

            buff_slice_high = self.filter_output_high[: len(freqs_low)]
            _create_pass_filters(
                buff_slice_high,
                freqs_high,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                mode="low",
            )

            return _fft_conv1d(samples, buff_slice_low - buff_slice_high)

        return samples
