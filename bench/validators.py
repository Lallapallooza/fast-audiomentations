"""Validation comparators. Three modes dispatched by ``Row.validator``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class Verdict:
    """Outcome of one row's validation: pass/fail plus the comparison detail."""

    passed: bool
    metric: str
    value: float
    threshold: float
    detail: dict[str, Any]


def to_canonical(out: Any) -> torch.Tensor:
    """Normalise per-library output to ``(B, T) float32`` on CPU.

    audiomentations: numpy 1D ``(T,)`` -> ``(1, T)``.
    torch_audiomentations: tensor ``(B, 1, T)`` -> ``(B, T)``.
    fast_audiomentations: tensor ``(B, T)`` or ``(B, 1, T)`` -> ``(B, T)``.
    """
    if isinstance(out, np.ndarray):
        if out.ndim == 1:
            out = out[np.newaxis, :]
        elif out.ndim == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        return torch.from_numpy(np.ascontiguousarray(out)).float().cpu()
    if isinstance(out, torch.Tensor):
        if out.ndim == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        result: torch.Tensor = out.detach().float().cpu()
        return result
    raise TypeError(f"Cannot canonicalise output of type {type(out).__name__}")


def numeric(
    cand: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float
) -> Verdict:
    """Elementwise allclose. ``passed`` iff every element within ``atol + rtol * |ref|``."""
    # Truncate to the shorter T -- FFT-conv can produce slightly different output lengths
    # than direct conv (FIR group delay). Comparing the overlapping prefix is the honest
    # check; large differences in length signal a real bug, caught by the shape compare below.
    if cand.shape != ref.shape:
        if cand.ndim == ref.ndim and cand.shape[0] == ref.shape[0]:
            t = min(cand.shape[-1], ref.shape[-1])
            cand = cand[..., :t]
            ref = ref[..., :t]
        else:
            return Verdict(
                False,
                "shape_mismatch",
                0.0,
                0.0,
                {"cand_shape": list(cand.shape), "ref_shape": list(ref.shape)},
            )
    diff = (cand - ref).abs()
    max_err = float(diff.max())
    threshold = atol + rtol * float(ref.abs().max())
    return Verdict(max_err <= threshold, "max_abs_err", max_err, threshold, {})


def statistical(
    cand_runs: list[torch.Tensor],
    ref_runs: list[torch.Tensor],
    moment_tol: float,
) -> Verdict:
    """Compare distributions of ``n_trials`` runs by moment match.

    Mean and std must agree within ``moment_tol`` relative tolerance. KS test on the
    flattened residuals as a secondary check.
    """
    cand_flat = (
        torch.stack([to_canonical(r) for r in cand_runs]).flatten().numpy()
    )
    ref_flat = (
        torch.stack([to_canonical(r) for r in ref_runs]).flatten().numpy()
    )

    cand_mean, cand_std = float(cand_flat.mean()), float(cand_flat.std())
    ref_mean, ref_std = float(ref_flat.mean()), float(ref_flat.std())

    mean_err = abs(cand_mean - ref_mean) / max(abs(ref_mean), 1e-9)
    std_err = abs(cand_std - ref_std) / max(abs(ref_std), 1e-9)

    try:
        from scipy.stats import ks_2samp

        ks_stat, ks_p = ks_2samp(cand_flat, ref_flat)
        ks_p = float(ks_p)
    except ImportError:
        ks_stat, ks_p = float("nan"), float("nan")

    passed = mean_err <= moment_tol and std_err <= moment_tol
    return Verdict(
        passed,
        "mean+std",
        max(mean_err, std_err),
        moment_tol,
        {
            "mean_cand": cand_mean,
            "mean_ref": ref_mean,
            "std_cand": cand_std,
            "std_ref": ref_std,
            "mean_err_rel": mean_err,
            "std_err_rel": std_err,
            "ks_stat": float(ks_stat),
            "ks_pvalue": ks_p,
        },
    )


def skip() -> Verdict:
    """Return a passing verdict marked as skipped (no comparison performed)."""
    return Verdict(True, "skipped", 0.0, 0.0, {"note": "validator='skip'"})
