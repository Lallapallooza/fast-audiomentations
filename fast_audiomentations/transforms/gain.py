from fast_audiomentations.transforms._impl._gain_triton import apply_gain as _apply_gain_triton

import random
import torch


class Gain:
    """
    Class for applying gain (volume adjustment) to audio samples.

    Attributes:
        min_gain_in_db (float): Minimum gain value in decibels.
        max_gain_in_db (float): Maximum gain value in decibels.
        p (float): Probability of applying the gain operation.
        dtype (torch.dtype): Data type for computation.
    """

    def __init__(self,
                 min_gain_in_db=-12,
                 max_gain_in_db=12,
                 p=0.5,
                 buffer_size=129,
                 dtype: torch.dtype = torch.float32):
        """
        Initializes the Gain class with given gain range, probability, and buffer size.

        @param min_gain_in_db: Minimum gain value in decibels.
        @param max_gain_in_db: Maximum gain value in decibels.
        @param p: Probability of applying the gain operation.
        @param buffer_size: Size of the buffer for random gain generation.
        @param dtype: Data type for computation.
        """
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        self.p = p
        self.dtype = dtype

        self.random_buffer = torch.empty(buffer_size, device='cuda', dtype=dtype)

    def __generate_random_amplitude_ratios(self, num_audios):
        """
        Generate random amplitude ratios for the gain operation.

        @param num_audios: Number of audio samples to process.
        @return: A tensor of random amplitude ratios.
        """
        assert num_audios <= self.random_buffer.size(0)

        slice = self.random_buffer[:num_audios]
        slice.uniform_(self.min_gain_in_db, self.max_gain_in_db)

        return slice

    def __call__(self, samples: torch.Tensor, sample_rate: int, inplace=False):
        """
        Apply gain (volume adjustment) to the audio samples.

        @param samples: Input audio samples tensor.
        @param sample_rate: Sample rate of the audio. Not used in this function but included for API consistency.
        @param inplace: If True, perform the operation in-place.
        @return: Audio samples after applying gain.
        """
        if random.random() < self.p:
            gain_factors = self.__generate_random_amplitude_ratios(samples.shape[0])
            return _apply_gain_triton(samples, gain_factors, inplace=inplace)
        return samples
