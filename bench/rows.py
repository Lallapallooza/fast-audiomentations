"""All audiomentations transforms registered as one-liner rows.

Format per row: ``Row(name, lib, ctor, [tolerances...])``.

Order roughly tracks importance for ASR / audio-ML pipelines (most-used first).
Adding a new transform is a literal append: copy a block, edit the names.
``ours`` rows are present only for transforms fast-audiomentations implements
today; missing ``ours`` rows are flagged as ``unimplemented`` by ``--mode list``.

Stochastic transforms set ``validator="statistical"`` and ``n_trials > 1``
because their output isn't bit-equal to a reference run.

For determinism in numeric mode, parameter ranges are pinned (``min == max``)
so the transform reduces to a fixed deterministic operation.
"""

from __future__ import annotations

from typing import Any

import audiomentations as am
import torch_audiomentations as ta

import fast_audiomentations as fa
from bench.registry import Row, add

# =============================================================================
# Tier 1: most-used in ASR / audio-ML training pipelines.
# =============================================================================

# --- Gain ---------------------------------------------------------------------
add(
    Row(
        "Gain", "am", lambda: am.Gain(min_gain_db=6.0, max_gain_db=6.0, p=1.0)
    ),
    Row(
        "Gain",
        "ta",
        lambda: ta.Gain(
            min_gain_in_db=5.999,
            max_gain_in_db=6.001,
            p=1.0,
            output_type="tensor",
        ),
        atol=1e-3,
        rtol=1e-3,
    ),
    Row(
        "Gain",
        "ours",
        lambda: fa.Gain(min_gain_in_db=6.0, max_gain_in_db=6.0, p=1.0),
    ),
)

# --- Clip ---------------------------------------------------------------------
add(
    Row("Clip", "am", lambda: am.Clip(a_min=-0.5, a_max=0.5, p=1.0)),
    Row("Clip", "ours", lambda: fa.Clip(min=-0.5, max=0.5, p=1.0)),
)

# --- PolarityInversion --------------------------------------------------------
add(
    Row("PolarityInversion", "am", lambda: am.PolarityInversion(p=1.0)),
    Row(
        "PolarityInversion",
        "ta",
        lambda: ta.PolarityInversion(p=1.0, output_type="tensor"),
    ),
    Row("PolarityInversion", "ours", lambda: fa.PolarityInversion(p=1.0)),
)

# --- Reverse ------------------------------------------------------------------
add(
    Row("Reverse", "am", lambda: am.Reverse(p=1.0)),
    Row("Reverse", "ours", lambda: fa.Reverse(p=1.0)),
)

# --- TimeInversion: torch_audiomentations equivalent of Reverse ---------------
# Pair under "Reverse" so it shares the audiomentations reference.
add(
    Row(
        "Reverse", "ta", lambda: ta.TimeInversion(p=1.0, output_type="tensor")
    ),
)

# --- AddGaussianNoise ---------------------------------------------------------
add(
    Row(
        "AddGaussianNoise",
        "am",
        lambda: am.AddGaussianNoise(
            min_amplitude=0.005, max_amplitude=0.005, p=1.0
        ),
        validator="statistical",
        n_trials=200,
        moment_tol=0.10,
    ),
    # ours: not yet implemented
)

# --- AddGaussianSNR -----------------------------------------------------------
add(
    Row(
        "AddGaussianSNR",
        "am",
        lambda: am.AddGaussianSNR(min_snr_db=20.0, max_snr_db=20.0, p=1.0),
        validator="statistical",
        n_trials=200,
        moment_tol=0.10,
    ),
)

# --- LowPassFilter ------------------------------------------------------------
# All three libs use different filter algorithms (am: scipy biquad IIR; ta: same;
# ours: FFT-based FIR). Statistical comparison: spectral content match, not
# elementwise. Magnitude attenuation in stop-band is the right notion.
add(
    Row(
        "LowPassFilter",
        "am",
        lambda: am.LowPassFilter(
            min_cutoff_freq=2000, max_cutoff_freq=2000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "LowPassFilter",
        "ta",
        lambda: ta.LowPassFilter(
            min_cutoff_freq=1999,
            max_cutoff_freq=2001,
            sample_rate=44100,
            p=1.0,
            output_type="tensor",
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "LowPassFilter",
        "ours",
        lambda: fa.LowPassFilter(
            min_cutoff_freq=2000, max_cutoff_freq=2000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
)

# --- HighPassFilter -----------------------------------------------------------
add(
    Row(
        "HighPassFilter",
        "am",
        lambda: am.HighPassFilter(
            min_cutoff_freq=500, max_cutoff_freq=500, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "HighPassFilter",
        "ta",
        lambda: ta.HighPassFilter(
            min_cutoff_freq=499,
            max_cutoff_freq=501,
            sample_rate=44100,
            p=1.0,
            output_type="tensor",
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "HighPassFilter",
        "ours",
        lambda: fa.HighPassFilter(
            min_cutoff_freq=500, max_cutoff_freq=500, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
)

# --- BandPassFilter -----------------------------------------------------------
add(
    Row(
        "BandPassFilter",
        "am",
        lambda: am.BandPassFilter(
            min_center_freq=1000, max_center_freq=1000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "BandPassFilter",
        "ta",
        lambda: ta.BandPassFilter(
            min_center_frequency=999,
            max_center_frequency=1001,
            sample_rate=44100,
            p=1.0,
            output_type="tensor",
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "BandPassFilter",
        "ours",
        lambda: fa.BandPassFilter(
            min_center_freq=1000, max_center_freq=1000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
)

# --- BandStopFilter -----------------------------------------------------------
add(
    Row(
        "BandStopFilter",
        "am",
        lambda: am.BandStopFilter(
            min_center_freq=1000, max_center_freq=1000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "BandStopFilter",
        "ta",
        lambda: ta.BandStopFilter(
            min_center_frequency=999,
            max_center_frequency=1001,
            sample_rate=44100,
            p=1.0,
            output_type="tensor",
        ),
        validator="skip",
        note="filter-class differences",
    ),
    Row(
        "BandStopFilter",
        "ours",
        lambda: fa.BandStopFilter(
            min_center_freq=1000, max_center_freq=1000, p=1.0
        ),
        validator="skip",
        note="filter-class differences",
    ),
)

# --- Normalize ---------------------------------------------------------------
# torch_audiomentations.PeakNormalization is the same op; pair under "Normalize".
add(
    Row("Normalize", "am", lambda: am.Normalize(p=1.0)),
    Row(
        "Normalize",
        "ta",
        lambda: ta.PeakNormalization(p=1.0, output_type="tensor"),
        atol=1e-3,
        rtol=1e-3,
    ),
)

# --- Shift --------------------------------------------------------------------
add(
    Row("Shift", "am", lambda: am.Shift(min_shift=0.1, max_shift=0.1, p=1.0)),
    Row(
        "Shift",
        "ta",
        lambda: ta.Shift(
            min_shift=0.099, max_shift=0.101, p=1.0, output_type="tensor"
        ),
        validator="skip",
        note="ta.Shift semantics differ from am.Shift",
    ),
    # ours: not yet implemented
)

# =============================================================================
# Tier 2: useful augmentations, no GPU baseline (ours = None today).
# =============================================================================


# --- AddBackgroundNoise -------------------------------------------------------
# Both libs need a noise corpus on disk. For ``ours``, the default torch
# DataLoader does per-call disk I/O which would dominate the measurement and
# bench the loader instead of the mix kernel. ``_CachedNoiseLoader`` pre-loads
# one noise file into a CPU tensor and yields the same (batch_size, T) pair on
# every ``next()``, isolating the kernel cost.
class _CachedNoiseLoader:
    """Infinite iterator yielding a fixed CUDA (batch, T) noise pair.

    Pre-loaded once on GPU. AddBackgroundNoise's ``_next_torch_batch`` does
    ``.to('cuda', non_blocking=True)``; that is a no-op on tensors already
    resident on CUDA, so the bench measures the mix kernel cost only, not
    a per-call host->device copy.
    """

    def __init__(self, noise_path: str, batch_size: int) -> None:
        import soundfile as sf
        import torch as _torch

        samples, _sr = sf.read(noise_path, dtype="float32")
        self._noise_batch = (
            _torch.from_numpy(samples)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .to("cuda")
        )
        self._lens = _torch.full(
            (batch_size,),
            self._noise_batch.shape[1],
            dtype=_torch.int64,
            device="cuda",
        )

    def __iter__(self) -> _CachedNoiseLoader:
        return self

    def __next__(self) -> tuple[Any, Any]:
        return self._noise_batch, self._lens


add(
    Row(
        "AddBackgroundNoise",
        "am",
        lambda: am.AddBackgroundNoise(sounds_path="tests/data", p=1.0),
        validator="skip",
        batches=(128,),
        note="stochastic noise selection",
    ),
    Row(
        "AddBackgroundNoise",
        "ours",
        lambda: fa.AddBackgroundNoise(
            noises_dataloader=_CachedNoiseLoader(
                "tests/data/44k_noise.wav", batch_size=128
            ),
            min_snr=10.0,
            max_snr=10.0,
            p=1.0,
            buffer_size=128,
        ),
        adapter="with_lens",
        validator="skip",
        batches=(128,),
        note="stochastic noise selection",
    ),
)

add(
    Row(
        "ClippingDistortion",
        "am",
        lambda: am.ClippingDistortion(
            min_percentile_threshold=10, max_percentile_threshold=10, p=1.0
        ),
    )
)
add(
    Row(
        "AddColorNoise",
        "am",
        lambda: am.AddColorNoise(
            min_snr_db=20.0,
            max_snr_db=20.0,
            min_f_decay=0.0,
            max_f_decay=0.0,
            p=1.0,
        ),
        validator="statistical",
        n_trials=200,
        moment_tol=0.15,
    )
)
add(
    Row(
        "AirAbsorption",
        "am",
        lambda: am.AirAbsorption(min_distance=10.0, max_distance=10.0, p=1.0),
    )
)
# Limiter requires audiomentations[extras]; not registered.
# LoudnessNormalization requires the loudness package; not registered.
add(
    Row(
        "TanhDistortion",
        "am",
        lambda: am.TanhDistortion(
            min_distortion=0.5, max_distortion=0.5, p=1.0
        ),
    )
)
add(
    Row(
        "BitCrush",
        "am",
        lambda: am.BitCrush(min_bit_depth=8, max_bit_depth=8, p=1.0),
    )
)
add(
    Row(
        "GainTransition",
        "am",
        lambda: am.GainTransition(
            min_gain_db=-12.0,
            max_gain_db=-12.0,
            min_duration=0.1,
            max_duration=0.1,
            p=1.0,
        ),
    )
)
add(
    Row(
        "Aliasing",
        "am",
        lambda: am.Aliasing(min_sample_rate=8000, max_sample_rate=8000, p=1.0),
    )
)
add(
    Row(
        "Padding",
        "am",
        lambda: am.Padding(
            mode="silence", min_fraction=0.1, max_fraction=0.1, p=1.0
        ),
    )
)
add(
    Row(
        "TimeMask",
        "am",
        lambda: am.TimeMask(min_band_part=0.05, max_band_part=0.05, p=1.0),
        validator="statistical",
        n_trials=50,
        moment_tol=0.15,
    )
)
add(
    Row(
        "RepeatPart",
        "am",
        lambda: am.RepeatPart(
            mode="replace",
            min_part_duration=0.1,
            max_part_duration=0.1,
            crossfade_duration=0.01,
            p=1.0,
        ),
        validator="skip",
        note="length-changing",
    )
)

# =============================================================================
# Tier 3: filters / EQ / specialty.
# =============================================================================

add(
    Row(
        "HighShelfFilter",
        "am",
        lambda: am.HighShelfFilter(
            min_center_freq=1500,
            max_center_freq=1500,
            min_gain_db=-6.0,
            max_gain_db=-6.0,
            p=1.0,
        ),
        atol=1e-2,
        rtol=1e-2,
    )
)
add(
    Row(
        "LowShelfFilter",
        "am",
        lambda: am.LowShelfFilter(
            min_center_freq=300,
            max_center_freq=300,
            min_gain_db=-6.0,
            max_gain_db=-6.0,
            p=1.0,
        ),
        atol=1e-2,
        rtol=1e-2,
    )
)
add(
    Row(
        "PeakingFilter",
        "am",
        lambda: am.PeakingFilter(
            min_center_freq=1000,
            max_center_freq=1000,
            min_gain_db=6.0,
            max_gain_db=6.0,
            min_q=1.0,
            max_q=1.0,
            p=1.0,
        ),
        atol=1e-2,
        rtol=1e-2,
    )
)
add(
    Row(
        "SevenBandParametricEQ",
        "am",
        lambda: am.SevenBandParametricEQ(
            min_gain_db=-6.0, max_gain_db=-6.0, p=1.0
        ),
        validator="statistical",
        n_trials=20,
        moment_tol=0.10,
    )
)

# =============================================================================
# Tier 4: complex / IO-heavy / codec-roundtrip (low priority for GPU port).
# =============================================================================

add(
    Row(
        "PitchShift",
        "am",
        lambda: am.PitchShift(min_semitones=2.0, max_semitones=2.0, p=1.0),
        atol=1e-1,
        rtol=1e-1,
    )
)
add(
    Row(
        "PitchShift",
        "ta",
        lambda: ta.PitchShift(
            min_transpose_semitones=1.99,
            max_transpose_semitones=2.01,
            sample_rate=48000,
            p=1.0,
            output_type="tensor",
        ),
        validator="skip",
        note="ta.PitchShift sample_rate constraints",
    )
)

add(
    Row(
        "TimeStretch",
        "am",
        lambda: am.TimeStretch(
            min_rate=1.1, max_rate=1.1, p=1.0, leave_length_unchanged=True
        ),
        atol=1e-1,
        rtol=1e-1,
    )
)

add(
    Row(
        "Resample",
        "am",
        lambda: am.Resample(
            min_sample_rate=22050, max_sample_rate=22050, p=1.0
        ),
        validator="skip",
        note="length-changing",
    )
)
add(
    Row(
        "AdjustDuration",
        "am",
        lambda: am.AdjustDuration(duration_seconds=2.0, p=1.0),
        validator="skip",
        note="length-changing",
    )
)

# =============================================================================
# Skipped (codec roundtrip / requires external assets / runtime-only):
#   - Mp3Compression: lossy codec; not a useful GPU target.
#   - ApplyImpulseResponse: requires IR files; out of scope for default fixture.
#   - AddShortNoises: requires noise corpus; same.
#   - RoomSimulator: long simulation; specialty.
#   - Trim: silence detection; numerical answer is "0 or 1 sample" -- noisy.
#   - Compose / OneOf / SomeOf / Lambda: containers, not transforms.
# =============================================================================
