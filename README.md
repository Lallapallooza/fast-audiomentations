# fast-audiomentations

> GPU-resident audio augmentations for PyTorch, dispatched as hand-written Triton kernels. Linux + CUDA only. MIT.

[![ci](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/ci.yml/badge.svg)](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/ci.yml)
[![release](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/release.yml/badge.svg)](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

---

## Contract

- **Linux x86_64 + CUDA only.** macOS / Windows / CPU-only / ROCm are out of scope. Every transform allocates its scratch buffers as `device='cuda'` in the constructor.
- **Batch-shaped tensors.** `samples` is a contiguous `(B, T)` `torch.Tensor`. `Clip` and `Gain` preserve `(B, T)`; the four FIR filters return `(B, 1, T_out)` because of the FFT-conv1d shape change.
- **No allocation on the hot path.** Every transform preallocates random / scratch / window / time buffers in `__init__`. The steady-state call is a `uniform_` fill plus a kernel launch.
- **Preallocated batch cap.** Random-parameter transforms ship `buffer_size=129`; runtime asserts `num_audios <= buffer_size`. Raise it in the constructor for larger batches.
- **No reproducibility plumbing.** `random.random()` and `tensor.uniform_()` draw from the global / CUDA RNG state without seed wiring.

## Install

This repo is consumed via [`uv`](https://github.com/astral-sh/uv). It is not on PyPI.

```bash
git clone https://github.com/Lallapallooza/fast-audiomentations
cd fast-audiomentations

uv sync                       # core deps (torch + triton)
uv sync --extra dali          # add nvidia-dali for AddBackgroundNoise.get_dali_dataloader
uv sync --extra bench         # add audiomentations / torch-audiomentations baselines
```

After `uv sync`, run `direnv allow` once. The repo's `.envrc` activates the venv on `cd`. On NixOS, also create a local-only `.env` with the two paths torch and Triton expect:

```bash
cat > .env <<'EOF'
LD_LIBRARY_PATH=/run/opengl-driver/lib
TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
EOF
```

`.env` is git-ignored so each contributor controls their own. The variables are no-ops on glibc distros (Ubuntu, Arch, Fedora). Without direnv, prefix commands with `uv run --env-file .env`.

Smoke test:

```bash
python -m examples.test       # prints SUCCESS
```

## Hello world

```python
import torch
from fast_audiomentations import Clip, LowPassFilter, AddBackgroundNoise

clip = Clip(min=-0.5, max=0.5, p=1.0)
lpf  = LowPassFilter(min_cutoff_freq=500, max_cutoff_freq=2000, p=1.0)

samples = torch.randn(8, 44100, device='cuda', dtype=torch.float32).contiguous()
clipped  = clip(samples, sample_rate=44100)          # (8, 44100)
filtered = lpf(samples, sample_rate=44100)           # (8, 1, T_out)
```

`AddBackgroundNoise` consumes a per-row noise corpus. The auto-picking factory uses NVIDIA DALI when installed, falls back to a torch `DataLoader` otherwise:

```python
from fast_audiomentations import AddBackgroundNoise

noise_dl = AddBackgroundNoise.get_dataloader(
    'path/to/noises',  # DALI: classification-folder root; torch: list of paths
    buffer_size=8,
    n_workers=2,
)
abn = AddBackgroundNoise(noises_dataloader=noise_dl, min_snr=-10, max_snr=10, p=1.0,
                          buffer_size=8)

audio = torch.randn(8, 44100, device='cuda', dtype=torch.float32).contiguous()
audio_lens = torch.full((8,), 44100, device='cuda')
mixed = abn(audio, audio_lens, sample_rate=44100)
```

## Transforms

| Class | Output shape | Notes |
|---|---|---|
| `Clip(min, max, p)` | `(B, T)` | Hard-clip every sample into `[min, max]`. |
| `Gain(min_gain_in_db, max_gain_in_db, p, buffer_size)` | `(B, T)` | Random per-row gain in dB. |
| `LowPassFilter(min_cutoff_freq, max_cutoff_freq, num_taps, buffer_size, p)` | `(B, 1, T_out)` | Random-cutoff FIR via FFT conv1d. |
| `HighPassFilter(...)` | `(B, 1, T_out)` | As above, high-pass. |
| `BandPassFilter(min_center_freq, max_center_freq, num_taps, buffer_size, p)` | `(B, 1, T_out)` | Random-bandwidth band-pass. |
| `BandStopFilter(...)` | `(B, 1, T_out)` | As above, band-stop. |
| `AddBackgroundNoise(noises_dataloader, min_snr, max_snr, p, buffer_size, dtype)` | `(B, T)` | Mixes a random noise row in at a per-row SNR. |

Every transform's `__call__` takes `(samples, sample_rate, inplace=False)`; `AddBackgroundNoise` adds a leading `samples_lens` tensor. `inplace` is honored by `Clip` / `Gain`; the FIR and mix paths cannot do in-place because they reshape or write to a new tensor.

## Versioning and releases

Versions follow [SemVer](https://semver.org). The active source of truth is `pyproject.toml` mirrored to `fast_audiomentations/__init__.py:__version__`.

`commitizen` inspects [Conventional Commits](https://www.conventionalcommits.org) since the last `v*` tag and decides:

- `feat:` -> minor bump
- `fix:` / `perf:` -> patch bump
- `BREAKING CHANGE:` footer or `!` after type -> major bump
- everything else -> no bump

`.github/workflows/release.yml` runs after CI passes on `main`: it calls `uv run cz bump --yes`, pushes the bump commit + tag, builds the wheel + sdist via `uv build`, and creates a GitHub release with notes generated from the commit log. If no commits since the last tag warrant a bump, the job exits cleanly.

To bump locally instead:

```bash
uv run cz bump --yes
git push --follow-tags
```

That pushes the tag manually; the `release-on-tag` job in the same workflow handles building artifacts and publishing the release.

## Benchmarks

A registry-style harness in `benchmark/` compares each transform against [`audiomentations`](https://github.com/iver56/audiomentations) (CPU) and [`torch-audiomentations`](https://github.com/asteroid-team/torch-audiomentations) (PyTorch on CUDA) across batch sizes `{1, 16, 32, 64, 128}` and `{float32, float16}` for `fast-audiomentations`.

```bash
uv sync --extra bench

# Edit benchmark/benchmark_data.py to point at your audio fixtures first.
python -m benchmark.run_all                   # full sweep
python -m benchmark.clip_benchmark            # one transform
```

Recorded results from the original 3090 Ti / i9-12900KF / 64 GB / Samsung 980 PRO host are in `benchmark_local_result/`. Numbers age out across hardware and library versions; treat them as a smoke test, not a contract. Reproduce on your own host before quoting.

## Development

```bash
uv sync                                          # installs the dev group too
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

uv run pre-commit run --all-files
uv run ruff check . && uv run ruff format .
uv run mypy
```

CI (`.github/workflows/ci.yml`) installs only `--only-group dev` and runs the same pipeline; it does not need torch / DALI / triton.

The `.git/hooks/pre-commit.legacy` slot (untracked) is a citor-style local-only hook for AI-tells unicode (em-dashes, smart quotes, arrows) that you opt into per clone. The hook is not part of the repo by design; install your own if you want it.

## Non-goals

- CPU fallback path. The transforms are GPU-only and that is the point.
- Reproducible RNG. Transforms draw from the global random / CUDA RNG state.
- Drop-in API parity with `audiomentations` or `torch-audiomentations`. Names and call signatures intentionally differ where the GPU-batched contract differs (see Contract above).
- `Compose` / `OneOf` orchestration wrappers. Not implemented yet.
- A PyPI release. Consume via `git clone` + `uv sync`.

## License

[MIT](LICENSE).
