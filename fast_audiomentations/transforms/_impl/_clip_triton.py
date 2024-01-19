import triton
import triton.language as tl
import torch
import itertools


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for (block_size, num_warps) in
        itertools.product([32, 64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32])
    ],
    key=['n_audios', 'audio_len'],
)
@triton.jit
def apply_clip_kernel(samples_ptr, min, max, output_ptr, n_audios, audio_len, BLOCK_SIZE: tl.constexpr):
    audio_idx = tl.program_id(0)

    if audio_idx >= n_audios:
        return

    for i in range(0, audio_len, BLOCK_SIZE):
        sample_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = sample_idx < audio_len

        samples = tl.load(samples_ptr + audio_idx * audio_len + sample_idx, mask=mask)
        result = tl.where(samples > max, max, samples)
        result = tl.where(result < min, min, result)
        tl.store(output_ptr + audio_idx * audio_len + sample_idx, result, mask=mask)


def apply_clip(samples: torch.Tensor, min: float, max: float, inplace: bool = False):
    assert min < max
    assert samples.ndim == 2

    n_audios, audio_len = samples.shape

    grid = lambda _: (n_audios,)

    if inplace:
        apply_clip_kernel[grid](samples, min, max, samples, n_audios, audio_len)
        return samples
    else:
        copy = torch.empty_like(samples, dtype=samples.dtype)
        apply_clip_kernel[grid](samples, min, max, copy, n_audios, audio_len)
        return copy
