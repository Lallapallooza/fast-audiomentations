# fast-audiomentations

> GPU-resident audio augmentations for PyTorch, dispatched as hand-written Triton kernels. Linux + CUDA only. MIT.

[![ci](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/ci.yml/badge.svg)](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/ci.yml)
[![release](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/release.yml/badge.svg)](https://github.com/Lallapallooza/fast-audiomentations/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

---

## Contents

- [Contract](#contract)
- [Install](#install)
  - [Add to a project (no clone)](#add-to-a-project-no-clone)
  - [Develop in this repo](#develop-in-this-repo)
- [Hello world](#hello-world)
- [Transforms](#transforms)
- [Benchmarks](#benchmarks)
- [Development](#development)
- [Versioning and releases](#versioning-and-releases)
- [Non-goals](#non-goals)
- [License](#license)

---

## Contract

- **Linux x86_64 + CUDA only.** macOS / Windows / CPU-only / ROCm are out of scope. Every transform allocates its scratch buffers as `device='cuda'` in the constructor.
- **Batch-shaped tensors.** `samples` is a contiguous `(B, T)` `torch.Tensor`. `Clip`, `Gain`, `PolarityInversion`, `Reverse`, and `AddBackgroundNoise` preserve `(B, T)`; the four FIR filters return `(B, 1, T_out)` because of the FFT-conv1d shape change.
- **No allocation on the hot path.** Every transform preallocates random / scratch / window / time buffers in `__init__`. The steady-state call is a `uniform_` fill plus a kernel launch.
- **Preallocated batch cap.** Random-parameter transforms ship `buffer_size=129`; runtime asserts `num_audios <= buffer_size`. Raise it in the constructor for larger batches.
- **No reproducibility plumbing.** `random.random()` and `tensor.uniform_()` draw from the global / CUDA RNG state without seed wiring.

## Install

Not on PyPI. Two paths.

### Add to a project (no clone)

```bash
# uv (recommended; reads [tool.uv] extra-index-url for the dali extra automatically)
uv add "fast-audiomentations @ git+https://github.com/Lallapallooza/fast-audiomentations"
uv add "fast-audiomentations[dali] @ git+https://github.com/Lallapallooza/fast-audiomentations"
uv add "fast-audiomentations @ git+https://github.com/Lallapallooza/fast-audiomentations@v0.1.0"

# pip (the [dali] extra needs the NVIDIA index spelled out)
pip install git+https://github.com/Lallapallooza/fast-audiomentations
pip install --extra-index-url https://pypi.nvidia.com \
    "fast-audiomentations[dali] @ git+https://github.com/Lallapallooza/fast-audiomentations"
```

### Develop in this repo

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
| `PolarityInversion(p)` | `(B, T)` | Negate every sample. |
| `Reverse(p)` | `(B, T)` | Mirror each row across time. |
| `LowPassFilter(min_cutoff_freq, max_cutoff_freq, num_taps, buffer_size, p)` | `(B, 1, T_out)` | Random-cutoff FIR via FFT conv1d. |
| `HighPassFilter(...)` | `(B, 1, T_out)` | As above, high-pass. |
| `BandPassFilter(min_center_freq, max_center_freq, num_taps, buffer_size, p)` | `(B, 1, T_out)` | Random-bandwidth band-pass. |
| `BandStopFilter(...)` | `(B, 1, T_out)` | As above, band-stop. |
| `AddBackgroundNoise(noises_dataloader, min_snr, max_snr, p, buffer_size, dtype)` | `(B, T)` | Mixes a random noise row in at a per-row SNR. |

Every transform's `__call__` takes `(samples, sample_rate, inplace=False)`; `AddBackgroundNoise` adds a leading `samples_lens` tensor. `inplace` is honored by `Clip`, `Gain`, and `PolarityInversion`; the FIR, mix, and reverse paths cannot run in-place because they reshape or need an out-of-place write pattern.

## Benchmarks

`bench/` is a single registry that drives both validation and benchmarking against `audiomentations` (CPU) and `torch-audiomentations` (CUDA). One CLI:

```bash
python -m bench --mode list                            # show every (transform, lib) cell
python -m bench --mode validate                        # numeric / statistical / skip per row
python -m bench --mode bench --batch 1,16,64,128 --dtype fp32,fp16
python -m bench --mode bench --filter Clip,Gain        # subset by transform name
```

Output goes to `results/<mode>-<git_sha>-<ts>.json`. Each record carries host invariants (GPU model, driver, torch + triton versions, lock hash, git sha) so cross-run comparisons stay honest.

Adding a transform is one tuple in `bench/rows.py`:

```python
add(
    Row("Clip", "am",   lambda: am.Clip(a_min=-0.5, a_max=0.5, p=1.0)),
    Row("Clip", "ours", lambda: fa.Clip(min=-0.5, max=0.5, p=1.0)),
)
```

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

## Versioning and releases

Conventional Commits. `feat:` bumps minor, `fix:` / `perf:` bump patch, `!` or `BREAKING CHANGE:` footer bump major; everything else does not bump. `.github/workflows/release.yml` runs `uv run cz bump --yes` after CI passes on `main`, pushes the bump commit and tag, builds via `uv build`, and creates the GitHub release.

To bump locally:

```bash
uv run cz bump --yes
git push --follow-tags
```

`commitizen.version_files` covers `pyproject.toml`, `fast_audiomentations/__init__.py:__version__`, and the install snippet in this README; the bump rewrites all three in lock-step.

## Non-goals

- CPU fallback path. Transforms are GPU-only and that is the point.
- Reproducible RNG. Transforms draw from the global / CUDA RNG without seed wiring.
- Drop-in API parity with `audiomentations` or `torch-audiomentations`. Names and call signatures intentionally differ where the GPU-batched contract differs.
- `Compose` / `OneOf` orchestration wrappers. Not implemented yet.

## License

[MIT](LICENSE).
