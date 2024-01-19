import triton
import triton.language as tl
import torch
import itertools


@triton.jit
def rms_kernel(audios, audios_real_lens, audios_max_len, batch_idx, BLOCK_SIZE_RMS: tl.constexpr):
    audios_real_lens_vals = tl.load(audios_real_lens + batch_idx)

    _mean = tl.zeros([BLOCK_SIZE_RMS], dtype=tl.float32)
    for offset in range(0, audios_max_len, BLOCK_SIZE_RMS):
        audios_block_ptr = offset + tl.arange(0, BLOCK_SIZE_RMS)
        audios_mask = audios_block_ptr < audios_real_lens_vals

        audios_vals = tl.load(audios + batch_idx * audios_max_len + audios_block_ptr, mask=audios_mask)
        audios_partial_sum_sq = tl.where(audios_mask, tl.math.pow(audios_vals, 2.0), 0)
        _mean += audios_partial_sum_sq

    audios_global_sum_sq = tl.sum(_mean, axis=0)
    return tl.sqrt(audios_global_sum_sq / audios_real_lens_vals)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_SUM': block_size_sum}, num_warps=num_warps)
        for (block_size_sum, num_warps) in
        itertools.product(
            [512, 1024],
            [2, 4, 8, 16]
        )
    ],
    key=['clean_audio_max_len', 'noisy_audio_max_len']
)
@triton.jit
def sum_with_snr_kernel(
        clean_audio, clean_audio_real_lens, clean_audio_max_len, desired_rms,
        noisy_audio_ptr, noisy_audio_real_lens, noisy_audio_max_len,
        output_ptr, BLOCK_SIZE_SUM: tl.constexpr, BLOCK_SIZE_RMS: tl.constexpr):
    batch_idx = tl.program_id(0)

    # RMS clean
    clean_audio_real_lens_val = tl.load(clean_audio_real_lens + batch_idx)
    clean_audio_rms = rms_kernel(clean_audio, clean_audio_real_lens, clean_audio_max_len, batch_idx, BLOCK_SIZE_RMS)

    # RMS noisy
    noisy_audio_real_lens_val = tl.load(noisy_audio_real_lens + batch_idx)

    noisy_audio_rms = rms_kernel(noisy_audio_ptr, noisy_audio_real_lens, noisy_audio_max_len, batch_idx, BLOCK_SIZE_RMS)

    # Desired RMS for noisy scale
    desired_rms_val = tl.load(desired_rms + batch_idx)
    relative_rms = clean_audio_rms / tl.math.pow(10.0, desired_rms_val / 20.0)

    for offset in range(0, clean_audio_max_len, BLOCK_SIZE_SUM):
        clean_audio_block_ptr = offset + tl.arange(0, BLOCK_SIZE_SUM)
        clean_audio_mask = clean_audio_block_ptr < clean_audio_real_lens_val
        clean_audio_vals = tl.load(
            clean_audio + batch_idx * clean_audio_max_len + clean_audio_block_ptr,
            mask=clean_audio_mask
        )

        """
        Adjusts the block's start position if it extends beyond the noisy audio array, shifting it leftward as needed.
        This adjustment keeps the data block within the noisy audio array limits, accounting for its circular nature

        Scenario without adjustment:
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                      |~~~~~~~~~~~~~~~~|
           (Block fits within the array, no adjustment needed)

        Scenario with adjustment:
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                                           |~~~~~~~~~~~~~~~~|
           (Block exceeds array bounds, needs to be shifted left)
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                                    |~~~~~~~~~~~~~~~~|  <--- Shifted left
        """
        offset_over_max = offset % noisy_audio_real_lens_val

        offset_adjusted = offset_over_max - tl.math.min(
            offset_over_max,
            tl.math.max(0, (offset_over_max + BLOCK_SIZE_SUM) - noisy_audio_real_lens_val)
        )

        noisy_audio_block_ptr = offset_adjusted + tl.arange(0, BLOCK_SIZE_SUM)

        noisy_audio_val = tl.load(
            noisy_audio_ptr + batch_idx * noisy_audio_max_len + noisy_audio_block_ptr,
            mask=noisy_audio_block_ptr < noisy_audio_real_lens_val
        )

        tl.store(
            output_ptr + batch_idx * clean_audio_max_len + clean_audio_block_ptr,
            clean_audio_vals + noisy_audio_val * (relative_rms / noisy_audio_rms),
            mask=clean_audio_mask
        )

def sum_with_snr_triton(samples: torch.Tensor, samples_lens: torch.Tensor, samples_noise, samples_noise_lens: torch.Tensor, snrs):
    assert samples.is_contiguous() and samples_noise.is_contiguous(), "Samples must be contiguous"

    B, T = samples.shape
    output = torch.empty_like(samples, device=samples.device, dtype=samples.dtype)

    grid = lambda opt: (B,)

    sum_with_snr_kernel[grid](
        samples, samples_lens, T, snrs,
        samples_noise, samples_noise_lens, samples_noise.shape[1],
        output, BLOCK_SIZE_RMS=max(1024, triton.next_power_of_2(max(T, samples_noise.shape[1]) // 1024)))

    return output