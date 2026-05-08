"""Triton kernel for polarity inversion: ``y[i] = -x[i]``."""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for (bs, nw) in itertools.product(
            [128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8]
        )
    ],
    key=["n_audios", "audio_len"],
)
@triton.jit
def apply_polarity_kernel(
    samples_ptr,
    output_ptr,
    n_audios,
    audio_len,
    BLOCK_SIZE: tl.constexpr,
):
    audio_idx = tl.program_id(0)
    if audio_idx >= n_audios:
        return

    for i in range(0, audio_len, BLOCK_SIZE):
        sample_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = sample_idx < audio_len
        offset = audio_idx * audio_len + sample_idx
        x = tl.load(samples_ptr + offset, mask=mask)
        tl.store(output_ptr + offset, -x, mask=mask)


def apply_polarity(
    samples: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    assert samples.ndim == 2
    n_audios, audio_len = samples.shape

    grid = lambda _: (n_audios,)
    if inplace:
        apply_polarity_kernel[grid](samples, samples, n_audios, audio_len)
        return samples
    out = torch.empty_like(samples)
    apply_polarity_kernel[grid](samples, out, n_audios, audio_len)
    return out
