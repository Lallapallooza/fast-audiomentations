import torch
import soundfile as sf

def load(path: str):
    samples, sr = sf.read(path, dtype="float32")
    return torch.tensor(samples, device='cuda', dtype=torch.float32).reshape(1, -1), sr

def save(samples: torch.Tensor, sr: int, path: str):
    if samples.ndim > 2:
        raise ValueError("Audios with more than 2 dimensions are not supported.")
    if samples.ndim == 2:
        if samples.shape[0] != 1:
            raise ValueError("Audios with shape (C, L) where C != 1 are not supported.")
        samples = samples.squeeze(0)

    sf.write(path, samples.cpu(), sr)
