"""Triton kernel for time reversal: ``y[i] = x[T - 1 - i]``."""

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
def apply_reverse_kernel(
    samples_ptr,
    output_ptr,
    n_audios,
    audio_len,
    BLOCK_SIZE: tl.constexpr,
):
    audio_idx = tl.program_id(0)
    if audio_idx >= n_audios:
        return

    base = audio_idx * audio_len
    last = audio_len - 1
    for i in range(0, audio_len, BLOCK_SIZE):
        sample_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = sample_idx < audio_len
        x = tl.load(samples_ptr + base + sample_idx, mask=mask)
        # Mirror the destination index across the row.
        dst = base + (last - sample_idx)
        tl.store(output_ptr + dst, x, mask=mask)


def apply_reverse(samples: torch.Tensor) -> torch.Tensor:
    # No inplace path: the kernel reads x[i] and writes y[T-1-i]; with output==samples
    # the early write would clobber a future read for the symmetric pair.
    assert samples.ndim == 2
    n_audios, audio_len = samples.shape

    grid = lambda _: (n_audios,)
    out = torch.empty_like(samples)
    apply_reverse_kernel[grid](samples, out, n_audios, audio_len)
    return out
