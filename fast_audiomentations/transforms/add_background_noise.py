import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from fast_audiomentations.transforms._impl._mix_triton import (
    sum_with_snr_triton as _sum_with_snr_triton,
)


class AddBackgroundNoise:
    """Mix random background-noise rows into the input batch at a random per-row SNR."""

    class AudioDataset(Dataset[tuple[torch.Tensor, int]]):
        """Mono-collapse a list of audio file paths into (waveform_1d, length) pairs."""

        def __init__(self, file_paths: list[str]) -> None:
            self.file_paths = file_paths

        def __len__(self) -> int:
            return len(self.file_paths)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            samples_np, _sr = sf.read(
                self.file_paths[idx % len(self.file_paths)], dtype="float32"
            )
            # Multichannel files collapse to first channel; matches the prior
            # torchaudio path (waveform[0]) and the kernel's mono assumption.
            if samples_np.ndim == 2:
                samples_np = samples_np[:, 0]
            waveform = torch.from_numpy(samples_np)
            return waveform, int(waveform.shape[0])

    def __init__(
        self,
        noises_dataloader: Any,
        min_snr: float = 15.0,
        max_snr: float = 20.0,
        p: float = 0.5,
        buffer_size: int = 129,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
        self.dtype = dtype
        self.noises_dataloader = noises_dataloader

        # Detect dataloader kind once. DALI pipelines expose `.run()`; torch
        # DataLoaders are iterables. The torch path keeps a long-lived iterator
        # so calls do not pay the per-call iter() cost.
        self._is_dali = hasattr(noises_dataloader, "run")
        self._dl_iter: Iterator[Any] | None = (
            None if self._is_dali else iter(noises_dataloader)
        )

        self.random_buffer = torch.empty(
            buffer_size, device="cuda", dtype=dtype
        )
        self.copy_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        self.padded_batch: torch.Tensor | None = None

    @staticmethod
    def get_default_dataloader(
        noises_paths: list[str],
        buffer_size: int = 128,
        n_workers: int = 8,
        prefetch_factor: int = 2,
    ) -> DataLoader[tuple[torch.Tensor, int]]:
        """Build a torch DataLoader that pads each batch to its longest row."""

        def audio_collate_fn(
            batch: list[tuple[torch.Tensor, int]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            max_length = max(waveform.size(0) for waveform, _ in batch)
            batch_waveforms: list[torch.Tensor] = []
            audio_lens: list[int] = []
            for waveform, waveform_len in batch:
                padded_waveform = torch.nn.functional.pad(
                    waveform,
                    (0, max_length - waveform.size(0)),
                    mode="constant",
                    value=0,
                )
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
            persistent_workers=True,
        )

    @staticmethod
    def get_dataloader(
        noises_paths: Any,
        buffer_size: int = 128,
        n_workers: int = 8,
        device_id: int = 0,
    ) -> Any:
        """Auto-pick: DALI pipeline if DALI is installed, else torch DataLoader.

        Both forms are accepted by the AddBackgroundNoise constructor; this is
        the recommended factory for callers that do not care which one is used.
        """
        try:
            return AddBackgroundNoise.get_dali_dataloader(
                noises_paths,
                buffer_size=buffer_size,
                n_workers=n_workers,
                device_id=device_id,
            )
        except ImportError:
            return AddBackgroundNoise.get_default_dataloader(
                noises_paths, buffer_size=buffer_size, n_workers=n_workers
            )

    @staticmethod
    def get_dali_dataloader(
        noises_paths: str,
        buffer_size: int = 128,
        n_workers: int = 8,
        device_id: int = 0,
    ) -> Any:
        """DALI pipeline yielding (audio, length) pairs from a classification-folder root.

        Requires the optional `dali` extra: `uv sync --extra dali` or
        `pip install fast-audiomentations[dali]`.
        """
        try:
            import nvidia.dali.fn as fn
            import nvidia.dali.types as types
            from nvidia.dali import pipeline_def
        except ImportError as e:
            raise ImportError(
                "nvidia-dali is required for AddBackgroundNoise.get_dali_dataloader; "
                "install via `uv sync --extra dali` or "
                "`pip install fast-audiomentations[dali]`"
            ) from e

        @pipeline_def  # type: ignore[untyped-decorator]
        def audio_pipeline(file_paths: str) -> tuple[Any, Any]:
            encoded, _ = fn.readers.file(file_root=file_paths)
            audio, _sr = fn.decoders.audio(
                encoded, dtype=types.FLOAT, downmix=True
            )
            audio_length = fn.shapes(audio)
            return audio, audio_length

        pipe = audio_pipeline(
            batch_size=buffer_size,
            num_threads=n_workers,
            device_id=device_id,
            file_paths=noises_paths,
            prefetch_queue_depth=2,
        )
        pipe.build()
        return pipe

    def __generate_random_snrs(self, num_audios: int) -> torch.Tensor:
        assert num_audios <= self.random_buffer.size(0)

        buff_slice = self.random_buffer[:num_audios]
        buff_slice.uniform_(self.min_snr, self.max_snr)

        return buff_slice

    def create_padded_batch(
        self, audios: tuple[Any, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack DALI's variable-length output into a (B, max_len) CUDA tensor + lengths."""
        # DALI's pipe.run() returns (TensorListGPU(audio), TensorListGPU(lengths)) of
        # variable-length rows. Pad them into a contiguous (B, max_len) CUDA matrix.
        audio_tensors = [np.array(tensor) for tensor in audios[0]]
        lengths = np.squeeze(np.array(audios[1]), 1)

        max_length = int(lengths.max())
        batch_size = len(audio_tensors)

        # TODO: some smart preallocation
        padded_batch = torch.zeros(batch_size, max_length, device="cuda")
        self.padded_batch = padded_batch

        for i, audio_tensor_np in enumerate(audio_tensors):
            audio_length = lengths[i]
            audio_tensor = torch.from_numpy(audio_tensor_np)
            padded_batch[i, :audio_length].copy_(
                audio_tensor, non_blocking=True
            )

        return padded_batch, torch.tensor(lengths, device="cuda")

    def _next_torch_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # torch DataLoader returns (stacked_padded_tensor, lengths_tensor) on CPU;
        # audio_collate_fn already pads to max_length within the batch. Cycle the
        # iterator on exhaustion so the loader behaves like an infinite stream.
        assert self._dl_iter is not None
        try:
            noises_cpu, lens_cpu = next(self._dl_iter)
        except StopIteration:
            self._dl_iter = iter(self.noises_dataloader)
            noises_cpu, lens_cpu = next(self._dl_iter)
        return (
            noises_cpu.to("cuda", non_blocking=True),
            lens_cpu.to("cuda", non_blocking=True),
        )

    def __call__(
        self,
        samples: torch.Tensor,
        samples_lens: torch.Tensor,
        sample_rate: int,  # noqa: ARG002 - kept for cross-transform API uniformity.
        inplace: bool = False,  # noqa: ARG002 - mix path returns a new tensor.
    ) -> torch.Tensor:
        """Mix a random noise row into each sample at a random per-row SNR with probability ``p``."""
        if random.random() < self.p:
            if self._is_dali:
                audios = self.noises_dataloader.run()
                noises, noises_lens = self.create_padded_batch(audios)
            else:
                noises, noises_lens = self._next_torch_batch()
            snrs = self.__generate_random_snrs(samples.shape[0])

            return _sum_with_snr_triton(
                samples, samples_lens, noises, noises_lens, snrs
            )
        return samples
