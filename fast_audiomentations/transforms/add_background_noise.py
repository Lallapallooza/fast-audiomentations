from fast_audiomentations.transforms._impl._mix_triton import sum_with_snr_triton as _sum_with_snr_triton

import random
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def audio_pipeline(file_paths):
    """
    Defines an NVIDIA DALI pipeline for audio file processing.

    @param file_paths: The root directory path where audio files are stored.
    @return: Processed audio and audio lengths as output from the pipeline.
    """
    encoded, _ = fn.readers.file(file_root=file_paths)
    audio, sr = fn.decoders.audio(encoded, dtype=types.FLOAT, downmix=True)
    audio_length = fn.shapes(audio)

    return audio, audio_length


class AddBackgroundNoise:
    """
    Class for adding background noise to audio samples.

    Attributes:
        noises_dataloader (DataLoader | DALIDataLoader): Dataloader for background noise samples.
        min_snr (float): Minimum Signal-to-Noise Ratio.
        max_snr (float): Maximum Signal-to-Noise Ratio.
        p (float): Probability of applying the augmentation.
        dtype (torch.dtype): Data type for computation.
    """
    class AudioDataset(Dataset):
        """
        Dataset class to handle loading of audio files.

        @param file_paths: List of paths to audio files.
        """
        def __init__(self, file_paths):
            self.file_paths = file_paths

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            """
            Get a single item from the dataset.

            @param idx: Index of the item.
            @return: Waveform and its length of the audio at the given index.
            """
            try:
                waveform, sample_rate = torchaudio.load(self.file_paths[idx % len(self.file_paths)])
                return waveform[0], waveform.shape[1]
            except Exception as e:
                print(e)

    def __init__(self,
                 noises_dataloader,
                 min_snr: float = 15.0,
                 max_snr: float = 20.0,
                 p=0.5,
                 buffer_size=129,
                 dtype: torch.dtype = torch.float32):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
        self.dtype = dtype
        self.noises_dataloader = noises_dataloader

        self.random_buffer = torch.empty(buffer_size, device='cuda', dtype=dtype)
        self.copy_stream = torch.cuda.Stream()
        self.padded_batch = None

    @staticmethod
    def get_default_dataloader(noises_paths, buffer_size=128, n_workers=8, prefetch_factor=2):
        """
        Create a default DataLoader for loading noise samples.

        @param noises_paths: List of paths to noise files.
        @param buffer_size: Size of the buffer for batch loading.
        @param n_workers: Number of worker threads for data loading.
        @param prefetch_factor: Number of batches to prefetch.
        @return: A DataLoader instance configured for loading noise samples.
        """
        def audio_collate_fn(batch):
            max_length = max(waveform.size(0) for waveform, _ in batch)
            batch_waveforms = []
            audio_lens = []
            for waveform, waveform_len in batch:
                padded_waveform = torch.nn.functional.pad(
                    waveform, (0, max_length - waveform.size(0)), mode='constant', value=0)
                batch_waveforms.append(padded_waveform)
                audio_lens.append(waveform_len)
            return torch.stack(batch_waveforms), torch.tensor(audio_lens)

        return DataLoader(
            AddBackgroundNoise.AudioDataset(noises_paths),
            batch_size=buffer_size,
            num_workers=n_workers,
            shuffle=True,
            prefetch_factor=prefetch_factor,
            pin_memory=False,
            collate_fn=audio_collate_fn,
            drop_last=True,
            persistent_workers=True
        )

    @staticmethod
    def get_dali_dataloader(noises_paths, buffer_size=128, n_workers=8, device_id=0):
        """
        Create a DataLoader using NVIDIA DALI for loading noise samples.

        @param noises_paths: List of paths to noise files.
        @param buffer_size: Size of the buffer for batch loading.
        @param n_workers: Number of worker threads for data loading.
        @param device_id: ID of the GPU device.
        @return: A DALI pipeline instance configured for loading noise samples.
        """
        pipe = audio_pipeline(batch_size=buffer_size, num_threads=n_workers, device_id=device_id,
                              file_paths=noises_paths, prefetch_queue_depth=2)
        pipe.build()
        return pipe


    def __generate_random_snrs(self, num_audios):
        assert num_audios <= self.random_buffer.size(0)

        buff_slice = self.random_buffer[:num_audios]
        buff_slice.uniform_(self.min_snr, self.max_snr)

        return buff_slice

    def create_padded_batch(self, audios):
        # Extract and convert audio tensors and lengths to NumPy arrays
        audio_tensors = [np.array(tensor) for tensor in audios[0]]
        lengths = np.squeeze(np.array(audios[1]), 1)  # Extracting and reshaping length array

        # Find the maximum length in the batch
        max_length = lengths.max()

        # Batch size is the number of audio tensors
        batch_size = len(audio_tensors)

        # Allocate a GPU matrix with the shape [batch_size, max_length]
        # TODO: some smart preallocation
        self.padded_batch = torch.zeros(batch_size, max_length, device='cuda')

        # Copy each audio tensor (converted from NumPy) to the padded batch
        for i, audio_tensor_np in enumerate(audio_tensors):
            audio_length = lengths[i]

            audio_tensor = torch.from_numpy(audio_tensor_np)

            self.padded_batch[i, :audio_length].copy_(audio_tensor, non_blocking=True)

        return self.padded_batch, torch.tensor(lengths, device='cuda')

    def __call__(self, samples: torch.Tensor, samples_lens: torch.Tensor, sample_rate: int, inplace=False):
        """
        Apply the add background noise transformation.

        @param samples: Input audio samples.
        @param samples_lens: Lengths of the audio samples.
        @param sample_rate: Sample rate of the audio.
        @param inplace: If True, perform operation in-place.
        @return: Augmented audio samples.
        """
        if random.random() < self.p:
            audios = self.noises_dataloader.run()
            noises, noises_lens = self.create_padded_batch(audios)
            snrs = self.__generate_random_snrs(samples.shape[0])

            return _sum_with_snr_triton(
                samples, samples_lens,
                noises, noises_lens,
                snrs
            )
        return samples