from fast_audiomentations.transforms._impl._clip_triton import apply_clip as _apply_clip_triton

import random
import torch


class Clip:
    """
    Class for applying clipping to audio samples.

    Attributes:
        __min (float): The minimum value for clipping.
        __max (float): The maximum value for clipping.
        p (float): Probability of applying the clipping operation.
    """
    def __init__(self, min: float = -1.0, max: float = 1.0, p=0.5):
        """
        Initializes the Clip class with given clipping range and probability.

        @param min: The minimum value for clipping.
        @param max: The maximum value for clipping.
        @param p: Probability of applying the clipping operation.
        """
        self.__min = min
        self.__max = max
        self.p = p

    def __call__(self, samples: torch.Tensor, sample_rate: int, inplace=False):
        """
        Apply clipping to the audio samples.

        @param samples: Input audio samples tensor.
        @param sample_rate: Sample rate of the audio. Not used in this function but included for API consistency.
        @param inplace: If True, perform the operation in-place.
        @return: Audio samples after applying clipping.
        """
        if random.random() < self.p:
            return _apply_clip_triton(samples, self.__min, self.__max, inplace=inplace)
        return samples