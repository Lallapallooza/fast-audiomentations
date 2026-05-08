"""Host invariants captured once per run for the JSON output."""

from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _uv_lock_sha256() -> str:
    p = Path("uv.lock")
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def collect() -> dict[str, Any]:
    """Read host invariants once. Safe to call without CUDA."""
    import torch

    cuda_ok = torch.cuda.is_available()
    return {
        "ts_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "uv_lock_sha256": _uv_lock_sha256(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "cpu-only",
        "triton": _pkg_version("triton"),
        "audiomentations": _pkg_version("audiomentations"),
        "torch_audiomentations": _pkg_version("torch-audiomentations"),
        "fast_audiomentations": _pkg_version("fast-audiomentations"),
        "gpu": torch.cuda.get_device_name(0) if cuda_ok else "none",
        "cuda_available": cuda_ok,
    }
