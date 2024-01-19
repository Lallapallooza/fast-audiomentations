from fast_audiomentations.transforms._impl._filter_triton import create_filters as _create_pass_filters
from fast_audiomentations.transforms._impl._filter_triton import fft_conv1d as _fft_conv1d

import random
import torch


class BandStopFilter:
    """
    Class for applying a band-stop filter to audio samples.

    Attributes:
        min_center_freq (int): Minimum center frequency for the band-stop filter.
        max_center_freq (int): Maximum center frequency for the band-stop filter.
        num_taps (int): Number of filter taps.
        buffer_size (int): Size of the buffer for processing.
        p (float): Probability of applying the augmentation.
    """
    def __init__(
            self,
            min_center_freq: int = 500,
            max_center_freq: int = 2000,
            num_taps: int = 101,
            buffer_size: int = 129,
            p: float = 0.5
    ):
        self.__min_center_freq = min_center_freq
        self.__max_center_freq = max_center_freq
        self.p = p
        self.num_taps = num_taps

        # Generating a Hamming window for filter design
        self.window = torch.hamming_window(num_taps, device='cuda', dtype=torch.float32, periodic=False)

        # Buffers for random frequency generation
        self.random_buffer_low = torch.empty(buffer_size, device='cuda')
        self.random_buffer_high = torch.empty(buffer_size, device='cuda')

        # Time buffer for filter design
        half = (num_taps - 1) // 2
        self.time = torch.arange(-half, half + 1, dtype=torch.float32, device='cuda')

        # Buffers for filter outputs
        self.filter_output_low = torch.empty((buffer_size, self.num_taps), device='cuda', dtype=torch.float32)
        self.filter_output_high = torch.empty((buffer_size, self.num_taps), device='cuda', dtype=torch.float32)

    def __generate_random_cutoffs(self, num_audios):
        """
        Generate random cutoff frequencies for the band-stop filter.

        @param num_audios: Number of audio samples to process.
        @return: Tuple of tensors containing low and high cutoff frequencies.
        """
        assert num_audios <= self.random_buffer_low.size(0)

        # Randomly generate low and high cutoff frequencies
        buff_slice_low = self.random_buffer_low[:num_audios]
        buff_slice_low.uniform_(self.__min_center_freq, self.__max_center_freq)

        buff_slice_high = self.random_buffer_high[:num_audios]
        buff_slice_high.uniform_(self.__min_center_freq, self.__max_center_freq)

        # Ensuring the high cutoff is always higher than the low cutoff
        # TODO: rewrite on triton as well
        buff_slice_high = torch.clip(buff_slice_high + buff_slice_low, 0, self.__max_center_freq)

        return buff_slice_low, buff_slice_high

    def __call__(self, samples: torch.Tensor, sample_rate: int, inplace=False):
        """
        Apply the band-stop filter to the given audio samples.

        @param samples: Input audio samples tensor.
        @param sample_rate: Sample rate of the audio.
        @param inplace: If True, perform operation in-place.
        @return: Audio samples after applying the band-stop filter.
        """
        if random.random() < self.p:
            # Generate random frequencies for the filter
            freqs_low, freqs_high = self.__generate_random_cutoffs(samples.shape[0])

            # Create filters based on the generated frequencies
            buff_slice_low = self.filter_output_low[:len(freqs_low)]
            _create_pass_filters(
                buff_slice_low,
                freqs_low,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                mode="low"
            )

            buff_slice_high = self.filter_output_high[:len(freqs_low)]
            _create_pass_filters(
                buff_slice_high,
                freqs_high,
                self.time,
                self.window,
                sample_rate,
                self.num_taps,
                mode="high"
            )

            # Apply the convolution with the created filters
            return _fft_conv1d(samples, buff_slice_low - buff_slice_high)

        return samples
