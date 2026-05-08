import soundfile as sf
import torch


def load(path: str) -> tuple[torch.Tensor, int]:
    """Read an audio file into a (1, L) CUDA float32 tensor and its sample rate."""
    samples, sr = sf.read(path, dtype="float32")
    waveform = torch.tensor(
        samples, device="cuda", dtype=torch.float32
    ).reshape(1, -1)
    return waveform, sr


def save(samples: torch.Tensor, sr: int, path: str) -> None:
    """Write a 1D or (1, L) tensor to ``path`` at the given sample rate."""
    if samples.ndim > 2:
        raise ValueError(
            "Audios with more than 2 dimensions are not supported."
        )
    if samples.ndim == 2:
        if samples.shape[0] != 1:
            raise ValueError(
                "Audios with shape (C, L) where C != 1 are not supported."
            )
        samples = samples.squeeze(0)

    sf.write(path, samples.cpu(), sr)
