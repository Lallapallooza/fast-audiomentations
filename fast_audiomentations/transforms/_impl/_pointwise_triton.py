"""Triton kernel for any subset of clip / gain / polarity in one launch.

Each op is gated by a ``tl.constexpr`` flag, so a single launch with one
flag set produces the same SASS as a dedicated single-op kernel and a
launch with multiple flags set fuses them into one load + N ops + one
store.
"""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
        for (block_size, num_warps) in itertools.product(
            [32, 64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32]
        )
    ],
    key=["n_audios", "audio_len"],
)
@triton.jit
def pointwise_kernel(
    samples_ptr,
    output_ptr,
    gain_ptr,
    clip_min,
    clip_max,
    n_audios,
    audio_len,
    HAS_GAIN: tl.constexpr,
    HAS_POLARITY: tl.constexpr,
    HAS_CLIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    audio_idx = tl.program_id(0)

    if audio_idx >= n_audios:
        return

    if HAS_GAIN:
        gain = tl.load(gain_ptr + audio_idx)

    for i in range(0, audio_len, BLOCK_SIZE):
        sample_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = sample_idx < audio_len
        offset = audio_idx * audio_len + sample_idx

        v = tl.load(samples_ptr + offset, mask=mask)
        if HAS_GAIN:
            v = v * gain
        if HAS_POLARITY:
            v = -v
        if HAS_CLIP:
            v = tl.minimum(tl.maximum(v, clip_min), clip_max)
        tl.store(output_ptr + offset, v, mask=mask)


def apply_pointwise(
    samples: torch.Tensor,
    gain: torch.Tensor | None,
    clip_min: float,
    clip_max: float,
    has_gain: bool,
    has_polarity: bool,
    has_clip: bool,
    inplace: bool = False,
) -> torch.Tensor:
    assert samples.ndim == 2

    n_audios, audio_len = samples.shape

    grid = lambda _: (n_audios,)

    output = samples if inplace else torch.empty_like(samples)
    # Disabled-gain branch is dead-coded by HAS_GAIN constexpr; the pointer
    # value is unused, so any valid CUDA pointer (samples itself) is safe.
    gain_ptr = gain if has_gain else samples

    pointwise_kernel[grid](
        samples,
        output,
        gain_ptr,
        clip_min,
        clip_max,
        n_audios,
        audio_len,
        HAS_GAIN=has_gain,
        HAS_POLARITY=has_polarity,
        HAS_CLIP=has_clip,
    )
    return output
