from fast_audiomentations.transforms._impl._filter_triton import create_filters as _create_low_pass_filters
from fast_audiomentations.transforms._impl._filter_triton import fft_conv1d as _fft_conv1d

import random
import torch


class LowPassFilter:
    """
    Class for applying a low-pass filter to audio samples.

    Attributes:
        min_cutoff_freq (int): Minimum cutoff frequency for the low-pass filter.
        max_cutoff_freq (int): Maximum cutoff frequency for the low-pass filter.
        num_taps (int): Number of filter taps.
        buffer_size (int): Size of the buffer for processing.
        p (float): Probability of applying the augmentation.
    """

    def __init__(
            self,
            min_cutoff_freq: int = 500,
            max_cutoff_freq: int = 2000,
            num_taps: int = 101,
            buffer_size: int = 129,
            p: float = 0.5
    ):
        self.__min_cutoff_freq = min_cutoff_freq
        self.__max_cutoff_freq = max_cutoff_freq
        self.p = p
        self.num_taps = num_taps
        self.window = torch.hamming_window(num_taps, device='cuda', dtype=torch.float32, periodic=False)
        self.random_buffer = torch.empty(buffer_size, device='cuda')
        half = (num_taps - 1) // 2
        self.time = torch.arange(-half, half + 1, dtype=torch.float32, device='cuda')
        self.filter_output = torch.empty((buffer_size, self.num_taps), device='cuda', dtype=torch.float32)

    def __generate_random_cutoffs(self, num_audios):
        """
        Generate random cutoff frequencies for the low-pass filter.

        @param num_audios: Number of audio samples to process.
        @return: A tensor of random cutoff frequencies.
        """
        assert num_audios <= self.random_buffer.size(0)

        buff_slice = self.random_buffer[:num_audios]
        buff_slice.uniform_(self.__min_cutoff_freq, self.__max_cutoff_freq)

        return buff_slice

    def __call__(self, samples: torch.Tensor, sample_rate: int, inplace=False):
        """
        Apply the low-pass filter to the audio samples.

        @param samples: Input audio samples tensor.
        @param sample_rate: Sample rate of the audio.
        @param inplace: If True, perform the operation in-place.
        @return: Audio samples after applying the low-pass filter.
        """
        if random.random() < self.p:
            freqs = self.__generate_random_cutoffs(samples.shape[0])

            buff_slice = self.filter_output[:len(freqs)]
            _create_low_pass_filters(
                buff_slice,
                freqs,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                "low"
            )
            return _fft_conv1d(samples, buff_slice)

        return samples