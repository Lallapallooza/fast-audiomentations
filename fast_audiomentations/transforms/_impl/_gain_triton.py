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
def apply_gain_kernel(samples_ptr, amplitude_ratios_ptr, output_ptr, n_audios, audio_len, BLOCK_SIZE: tl.constexpr):
    audio_idx = tl.program_id(0)

    if audio_idx >= n_audios:
        return

    gain = tl.load(amplitude_ratios_ptr + audio_idx)

    for i in range(0, audio_len, BLOCK_SIZE):
        sample_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = sample_idx < audio_len
        samples = tl.load(samples_ptr + audio_idx * audio_len + sample_idx, mask=mask)
        result = samples * gain
        tl.store(output_ptr + audio_idx * audio_len + sample_idx, result, mask=mask)


def apply_gain(samples: torch.Tensor, amplitude_ratios: torch.Tensor, inplace: bool = False):
    assert samples.ndim == 2 and amplitude_ratios.ndim == 1
    n_audios, audio_len = samples.shape

    grid = lambda _: (n_audios,)

    if inplace:
        apply_gain_kernel[grid](samples, amplitude_ratios, samples, n_audios, audio_len)
        return samples
    else:
        copy = torch.empty_like(samples, device='cuda', dtype=samples.dtype)
        apply_gain_kernel[grid](samples, amplitude_ratios, copy, n_audios, audio_len)
        return copy
