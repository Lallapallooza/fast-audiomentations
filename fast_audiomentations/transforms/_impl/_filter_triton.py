import torch

import triton
import triton.language as tl

import itertools
from scipy.signal import firwin
import numpy as np
import math
import soundfile as sf


import torch
import torch.nn.functional as F


@triton.jit
def sinc_kernel(
        output_ptr,
        cutoffs_ptr,
        indices_ptr,
        num_taps,
        window_ptr,
        half_sample_rate,
        mode: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(1)
    pos = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < num_taps

    cutoff_val = tl.load(cutoffs_ptr + batch_idx) / half_sample_rate
    index_val = tl.load(indices_ptr + pos, mask=mask)
    window_val = tl.load(window_ptr + pos, mask=mask)

    x = index_val * math.pi * cutoff_val
    sinc_val = tl.where(index_val == 0, 1., tl.sin(x) / x)
    windowed_sinc = sinc_val * window_val

    # Normalize each filter by the sum of its windowed sinc values
    normalized_sinc = windowed_sinc / tl.sum(windowed_sinc, axis=0)
    if mode == "high":
        center_idx = num_taps // 2
        adjusted_val = tl.where(pos == center_idx, 1.0 - normalized_sinc, -normalized_sinc)

        tl.store(output_ptr + batch_idx * num_taps + pos, adjusted_val, mask=mask)
    elif mode == "low":
        tl.store(output_ptr + batch_idx * num_taps + pos, normalized_sinc, mask=mask)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_filters(filter_output, cutoff_freqs, time, window, sample_rate, num_taps, mode):
    grid_size = (1, len(cutoff_freqs))

    sinc_kernel[grid_size](
        filter_output,
        cutoff_freqs,
        time,
        num_taps,
        window,
        0.5 * sample_rate,
        mode,
        triton.next_power_of_2(num_taps)
    )

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for (block_size, num_warps) in
        itertools.product([32, 64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32])
    ],
    key=['length', 'kernel_size', 'stride', 'n_frames']
)
@triton.jit
def unfold_kernel(input_ptr, output_ptr, length, kernel_size, stride, n_frames, BLOCK_SIZE: tl.constexpr):
    # Compute indices
    batch_idx = tl.program_id(0)

    # Global frame index
    frame_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Bounds check for the frame index
    mask = frame_idx < n_frames

    # Calculate position in input for each thread
    input_pos = frame_idx * stride

    # Each thread processes one frame if within bounds
    for i in range(kernel_size):
        in_bounds = mask & ((input_pos + i) < length)

        # Use tl.where to handle in-bounds and out-of-bounds cases
        val = tl.where(in_bounds, tl.load(input_ptr + batch_idx * length + input_pos + i, mask=in_bounds), 0)

        out_idx = batch_idx * n_frames * kernel_size + frame_idx * kernel_size + i
        tl.store(output_ptr + out_idx, val, mask=in_bounds)


def unfold_triton(input, kernel_size, stride):
    assert input.ndim >= 2, "Input tensor must be at least 2D"
    length = input.shape[-1]
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1

    # Prepare output tensor
    output_shape = list(input.shape)[:-1] + [n_frames, kernel_size]
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Grid dimensions
    grid = lambda META: (
        input.shape[0],
        triton.cdiv(n_frames, META['BLOCK_SIZE']) + (n_frames % META['BLOCK_SIZE'] != 0)
    )

    # Launch kernel
    unfold_kernel[grid](input, output, length, kernel_size, stride, n_frames)

    return output

def complex_mul_conjugate(a: torch.Tensor, b: torch.Tensor):
    a_real = a[..., 0].contiguous()
    a_imag = a[..., 1].contiguous()
    b_real = b[..., 0].contiguous()
    b_imag = b[..., 1].contiguous()

    return torch.stack(complex_mul_conjugate_triton(a_real, b_real, a_imag, b_imag), dim=-1)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for (num_warps) in [1, 2, 4, 8, 16, 32]
    ],
    key=['num_batches', 'num_frames', 'fft_size']
)
@triton.jit
def complex_mul_conjugate_kernel(
        a_real_ptr,
        b_real_ptr,
        a_imag_ptr,
        b_imag_ptr,
        output1_ptr,
        output2_ptr,
        num_batches,
        num_frames,
        fft_size,
        BLOCK_SIZE: tl.constexpr):
    # Compute indices for batch and fft
    batch_idx = tl.program_id(0)

    # Ensure we don't go out of bounds for batch index
    if batch_idx >= num_batches:
        return

    fft_idx = tl.arange(0, BLOCK_SIZE)
    fft_mask = fft_idx < fft_size

    batch_by_fft = batch_idx * fft_size

    b_real_val = tl.load(b_real_ptr + batch_by_fft + fft_idx, mask=fft_mask)
    b_imag_val = tl.load(b_imag_ptr + batch_by_fft + fft_idx, mask=fft_mask)

    for frame_idx in range(num_frames):
        global_idx = num_frames * batch_by_fft + frame_idx * fft_size + fft_idx

        a_real_val = tl.load(a_real_ptr + global_idx, mask=fft_mask)
        a_imag_val = tl.load(a_imag_ptr + global_idx, mask=fft_mask)

        result1 = a_real_val * b_real_val + a_imag_val * b_imag_val
        result2 = a_imag_val * b_real_val - a_real_val * b_imag_val

        tl.store(output1_ptr + global_idx, result1, mask=fft_mask)
        tl.store(output2_ptr + global_idx, result2, mask=fft_mask)


def complex_mul_conjugate_triton(a_real, b_real, a_imag, b_imag):
    assert a_real.shape[-1] == b_real.shape[-1]  # Ensure last dimensions match for multiplication

    num_batches, num_frames, fft_size = a_real.shape

    # Output tensor
    output1 = torch.empty_like(a_real)
    output2 = torch.empty_like(a_real)

    # Define grid size for the kernel launch
    grid_size = (num_batches,)

    # Launch the kernel

    complex_mul_conjugate_kernel[grid_size](
        a_real,
        b_real,
        a_imag,
        b_imag,
        output1,
        output2,
        num_batches,
        num_frames,
        fft_size,
        triton.next_power_of_2(fft_size)
    )

    return output1, output2


def fft_conv1d(
        x: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 0,
        block_ratio: float = 8):
    x = F.pad(x, (padding, padding))
    B, L = x.shape
    _, kernel_size = weight.shape

    block_size: int = min(int(kernel_size * block_ratio), L)

    weight = F.pad(weight, (0, block_size - weight.shape[-1]), mode='constant', value=0)
    if weight.dtype != torch.float16 and weight.shape[1].bit_count() != 1:
        weight_z = torch.view_as_real(torch.fft.rfft(weight.to(torch.float32), dim=-1)).to(torch.float16)
    else:
        weight_z = torch.view_as_real(torch.fft.rfft(weight, dim=-1))

    frames = unfold_triton(x, block_size, block_size - kernel_size + 1)

    if frames.dtype == torch.float16 and frames.shape[1].bit_count() != 1:
        frames_z = torch.view_as_real(torch.fft.rfft(frames.to(torch.float32), dim=-1)).to(torch.float16)
    else:
        frames_z = torch.view_as_real(torch.fft.rfft(frames, dim=-1))

    out_z = complex_mul_conjugate(frames_z, weight_z)

    if out_z.dtype == torch.float16 and out_z.shape[1].bit_count() != 1:
        out = torch.fft.irfft(torch.view_as_complex(out_z.to(torch.float32)), block_size, dim=-1).to(torch.float16)
    else:
        out = torch.fft.irfft(torch.view_as_complex(out_z), block_size, dim=-1)

    out = out[..., :-kernel_size + 1]
    out = out.reshape(B, 1, -1)
    out = out[..., ::stride]
    target_length = (L - kernel_size) // stride + 1
    out = out[..., :target_length]

    return out