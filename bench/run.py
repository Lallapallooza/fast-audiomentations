"""Validate + benchmark dispatch driver.

Usage:
    python -m bench --mode validate [--filter Clip,Gain] [--out results/v.json]
    python -m bench --mode bench    [--filter ...] [--batch 1,16,64,128]
                                    [--dtype fp32,fp16] [--iters 200]
                                    [--warmup 25] [--out results/b.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench import host, validators
from bench.registry import ROWS, Row

# Fixture: one wav loaded once, repeated to fill the batch.
_FIXTURE_WAV = Path("tests/data/44k.wav")
_SAMPLE_RATE = 44100


def _load_fixture(batch: int, dtype: str) -> tuple[np.ndarray, torch.Tensor]:
    import soundfile as sf

    samples, sr = sf.read(str(_FIXTURE_WAV), dtype="float32")
    if sr != _SAMPLE_RATE:
        raise RuntimeError(
            f"Fixture sample rate {sr} != expected {_SAMPLE_RATE}"
        )
    np_x = np.tile(samples, (batch, 1))  # (B, T) float32 numpy
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype]
    t_x = (
        torch.from_numpy(np_x)
        .to(device="cuda", dtype=torch_dtype)
        .contiguous()
    )
    return np_x, t_x


# --- per-library call adapters -------------------------------------------------


def call_am(transform: Any, x_np: np.ndarray) -> Any:
    """audiomentations: numpy 1D in, 1D out. Run row-by-row."""
    out = np.empty_like(x_np)
    for i in range(x_np.shape[0]):
        out[i] = transform(samples=x_np[i], sample_rate=_SAMPLE_RATE)
    return out


def call_ta(transform: Any, x_t: torch.Tensor) -> Any:
    """torch_audiomentations: (B, C, T) tensor in/out."""
    x_bct = x_t.unsqueeze(1)  # (B, 1, T)
    out = transform(samples=x_bct, sample_rate=_SAMPLE_RATE)
    if hasattr(out, "samples"):  # ObjectDict in newer versions
        out = out.samples
    return out


def call_ours(transform: Any, x_t: torch.Tensor) -> Any:
    """fast_audiomentations: (B, T) tensor in/out."""
    return transform(samples=x_t, sample_rate=_SAMPLE_RATE)


def call_ours_with_lens(transform: Any, x_t: torch.Tensor) -> Any:
    """fast_audiomentations.AddBackgroundNoise needs ``samples_lens``."""
    lens = torch.full(
        (x_t.shape[0],), x_t.shape[1], device="cuda", dtype=torch.int64
    )
    return transform(samples=x_t, samples_lens=lens, sample_rate=_SAMPLE_RATE)


_CALL: dict[str, Callable[[Any, Any], Any]] = {
    "am": call_am,
    "ta": call_ta,
    "ours": call_ours,
    "ours_with_lens": call_ours_with_lens,
}


def _adapter_key(row: Row) -> str:
    if row.lib == "ours" and row.adapter == "with_lens":
        return "ours_with_lens"
    return row.lib


def _input_for(row: Row, x_np: np.ndarray, x_t: torch.Tensor) -> Any:
    return x_np if row.lib == "am" else x_t


# --- mode: validate -----------------------------------------------------------


def _run_validate(
    rows: list[Row],
    batch: int,
) -> list[dict[str, Any]]:
    """Group rows by ``name``; ``am`` row is the reference; diff every other row."""
    x_np, x_t = _load_fixture(batch, "fp32")
    out: list[dict[str, Any]] = []

    by_name: dict[str, list[Row]] = {}
    for r in rows:
        by_name.setdefault(r.name, []).append(r)

    for name, group in by_name.items():
        am_rows = [r for r in group if r.lib == "am"]
        if not am_rows:
            for r in group:
                out.append(  # noqa: PERF401 - body too rich for list-extend rewrite
                    {
                        "mode": "validate",
                        "name": name,
                        "lib": r.lib,
                        "batch": batch,
                        "passed": False,
                        "metric": "no_reference",
                        "value": 0.0,
                        "threshold": 0.0,
                        "detail": {
                            "note": "no audiomentations row to compare against"
                        },
                    }
                )
            continue
        am_row = am_rows[0]

        # Skip reference invocation entirely when every candidate in the group
        # has validator="skip" -- the am call is not used and may not be
        # invocable (e.g. length-changing transforms break the fixture loop).
        candidates = [r for r in group if r is not am_row]
        needs_reference = any(c.validator != "skip" for c in candidates)
        if not needs_reference:
            for r in group:
                v = validators.skip() if r is not am_row else None
                out.append(
                    {
                        "mode": "validate",
                        "name": name,
                        "lib": r.lib,
                        "batch": batch,
                        "passed": True,
                        "metric": "reference" if r is am_row else "skipped",
                        "value": 0.0,
                        "threshold": 0.0,
                        "detail": v.detail if v else {},
                    }
                )
            continue

        n_trials = max(
            (r.n_trials for r in candidates if r.validator == "statistical"),
            default=1,
        )
        ref_runs = [
            _invoke_safely(am_row, _input_for(am_row, x_np, x_t))
            for _ in range(n_trials)
        ]
        if any(r is None for r in ref_runs):
            out.append(
                _error_record(
                    am_row, batch, "validate", "ref_invocation_failed"
                )
            )
            continue

        for r in group:
            if r is am_row:
                out.append(
                    {
                        "mode": "validate",
                        "name": name,
                        "lib": "am",
                        "batch": batch,
                        "passed": True,
                        "metric": "reference",
                        "value": 0.0,
                        "threshold": 0.0,
                        "detail": {},
                    }
                )
                continue

            if r.validator == "skip":
                v = validators.skip()
            elif r.validator == "statistical":
                cand_runs = [
                    _invoke_safely(r, _input_for(r, x_np, x_t))
                    for _ in range(r.n_trials)
                ]
                if any(c is None for c in cand_runs):
                    out.append(
                        _error_record(
                            r, batch, "validate", "cand_invocation_failed"
                        )
                    )
                    continue
                # All entries are non-None after the guard above; cast for mypy.
                cand_tensors = [validators.to_canonical(c) for c in cand_runs]
                ref_tensors = [validators.to_canonical(rr) for rr in ref_runs]
                v = validators.statistical(
                    cand_tensors, ref_tensors, r.moment_tol
                )
            else:
                cand = _invoke_safely(r, _input_for(r, x_np, x_t))
                if cand is None:
                    out.append(
                        _error_record(
                            r, batch, "validate", "cand_invocation_failed"
                        )
                    )
                    continue
                v = validators.numeric(
                    validators.to_canonical(cand),
                    validators.to_canonical(ref_runs[0]),
                    r.atol,
                    r.rtol,
                )

            out.append(
                {
                    "mode": "validate",
                    "name": name,
                    "lib": r.lib,
                    "batch": batch,
                    "passed": v.passed,
                    "metric": v.metric,
                    "value": v.value,
                    "threshold": v.threshold,
                    "detail": v.detail,
                }
            )
    return out


def _invoke_safely(row: Row, x: Any) -> Any | None:
    try:
        transform = row.ctor()
        return _CALL[_adapter_key(row)](transform, x)
    except Exception as e:
        row_repr = f"{row.name}/{row.lib}"
        print(
            f"  [error] {row_repr}: {type(e).__name__}: {e}", file=sys.stderr
        )
        return None


def _error_record(
    row: Row, batch: int, mode: str, kind: str
) -> dict[str, Any]:
    return {
        "mode": mode,
        "name": row.name,
        "lib": row.lib,
        "batch": batch,
        "passed": False,
        "metric": kind,
        "value": 0.0,
        "threshold": 0.0,
        "detail": {"note": kind},
    }


# --- mode: bench --------------------------------------------------------------


def _bench_one(
    row: Row,
    x_np: np.ndarray,
    x_t: torch.Tensor,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    transform = row.ctor()
    fn_call = _CALL[_adapter_key(row)]
    x = _input_for(row, x_np, x_t)
    use_cuda_timer = row.lib in ("ours", "ta") and torch.cuda.is_available()

    if use_cuda_timer:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        fn_call(transform, x)
    if use_cuda_timer:
        torch.cuda.synchronize()

    timings_us: list[float] = []
    if use_cuda_timer:
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            e = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            s.record()
            fn_call(transform, x)
            e.record()
            e.synchronize()
            timings_us.append(s.elapsed_time(e) * 1000.0)  # ms -> us
        peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn_call(transform, x)
            t1 = time.perf_counter_ns()
            timings_us.append((t1 - t0) / 1000.0)
        peak_mem_mib = 0.0

    arr = np.asarray(timings_us)
    return {
        "median_us": float(np.median(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_us": float(arr.mean()),
        "peak_mem_mib": peak_mem_mib,
        "iters": iters,
    }


def _run_bench(
    rows: list[Row],
    batches: list[int],
    dtypes: list[str],
    iters: int,
    warmup: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in batches:
        for dtype in dtypes:
            try:
                x_np, x_t = _load_fixture(batch, dtype)
            except Exception as e:
                print(
                    f"  [error] fixture batch={batch} dtype={dtype}: {e}",
                    file=sys.stderr,
                )
                continue
            for r in rows:
                if r.lib == "am" and dtype == "fp16":
                    continue  # audiomentations is float32-only
                if r.batches is not None and batch not in r.batches:
                    continue
                try:
                    stats = _bench_one(
                        r, x_np, x_t, iters=iters, warmup=warmup
                    )
                    out.append(
                        {
                            "mode": "bench",
                            "name": r.name,
                            "lib": r.lib,
                            "batch": batch,
                            "dtype": dtype,
                            **stats,
                        }
                    )
                    print(
                        f"  {r.name:<24} b={batch:<4} {dtype:<5} "
                        f"{r.lib:<6} median={stats['median_us']:8.2f}us  "
                        f"p95={stats['p95_us']:8.2f}us",
                        flush=True,
                    )
                except Exception as e:
                    err = _error_record(
                        r, batch, "bench", f"{type(e).__name__}: {e}"
                    )
                    err["dtype"] = dtype
                    out.append(err)
                    print(
                        f"  {r.name:<24} b={batch:<4} {dtype:<5} "
                        f"{r.lib:<6} FAIL: {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
    return out


# --- main ---------------------------------------------------------------------


def _import_rows() -> None:
    """Import the rows module so its module-level ``add(...)`` calls register."""
    import bench.rows  # noqa: F401


def _filter_rows(rows: list[Row], names: str | None) -> list[Row]:
    if not names:
        return list(rows)
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    return [r for r in rows if r.name in wanted]


def main() -> int:
    """CLI entry point. See module docstring for usage."""
    parser = argparse.ArgumentParser(prog="bench")
    parser.add_argument(
        "--mode", choices=["validate", "bench", "list"], required=True
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated transform names. Default: all.",
    )
    parser.add_argument(
        "--batch",
        default="64",
        help="Comma-separated batch sizes (bench only).",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        help="Comma-separated dtypes (bench only): fp32,fp16.",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write JSON. Default: results/<mode>-<git-sha>-<ts>.json",
    )
    args = parser.parse_args()

    _import_rows()
    rows = _filter_rows(ROWS, args.filter)
    if not rows:
        print("No rows match the filter.", file=sys.stderr)
        return 2

    if args.mode == "list":
        return _run_list()

    h = host.collect()

    if args.mode == "validate":
        results = _run_validate(rows, batch=8)
    else:
        batches = [int(b) for b in args.batch.split(",")]
        dtypes = [d.strip() for d in args.dtype.split(",")]
        results = _run_bench(
            rows, batches, dtypes, iters=args.iters, warmup=args.warmup
        )

    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "host": h,
        "results": results,
    }
    out_path = _resolve_out(args.out, args.mode, h["git_sha"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    _print_summary(args.mode, results)
    print(f"\nWrote {out_path} ({len(results)} records).")

    if args.mode == "validate" and any(not r["passed"] for r in results):
        return 1
    return 0


def _run_list() -> int:
    """Print rows + which transforms have which libs implemented."""
    _import_rows()
    by_name: dict[str, set[str]] = {}
    for r in ROWS:
        by_name.setdefault(r.name, set()).add(r.lib)

    am_total = sum(1 for libs in by_name.values() if "am" in libs)
    ta_total = sum(1 for libs in by_name.values() if "ta" in libs)
    ours_total = sum(1 for libs in by_name.values() if "ours" in libs)

    print(f"\n{'transform':<28} {'am':<6}{'ta':<6}{'ours':<6}")
    print("-" * 48)
    for name in sorted(by_name):
        libs = by_name[name]
        print(
            f"{name:<28} "
            f"{'X' if 'am' in libs else '.':<6}"
            f"{'X' if 'ta' in libs else '.':<6}"
            f"{'X' if 'ours' in libs else '.':<6}"
        )
    print("-" * 48)
    print(f"{'TOTAL':<28} {am_total:<6}{ta_total:<6}{ours_total:<6}")
    print(
        f"\nUnimplemented in fast-audiomentations: "
        f"{am_total - ours_total} transforms."
    )
    return 0


def _resolve_out(out_arg: str | None, mode: str, git_sha: str) -> Path:
    if out_arg:
        return Path(out_arg)
    ts = int(time.time())
    return Path("results") / f"{mode}-{git_sha[:8]}-{ts}.json"


def _print_summary(mode: str, results: list[dict[str, Any]]) -> None:
    if mode == "validate":
        passed = sum(1 for r in results if r["passed"])
        failed = sum(1 for r in results if not r["passed"])
        print(
            f"\nvalidate: {passed} passed, {failed} failed (of {len(results)} rows)"
        )
        for r in results:
            if not r["passed"]:
                print(
                    f"  FAIL {r['name']}/{r['lib']:<6} {r['metric']}={r['value']:.4g} "
                    f"thr={r['threshold']:.4g} {r['detail']}"
                )
    else:
        # Group by (name, batch, dtype); for each group show speedup of ours vs am.
        by_cell: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
        for r in results:
            if "median_us" not in r:
                continue
            key = (r["name"], r["batch"], r["dtype"])
            by_cell.setdefault(key, {})[r["lib"]] = r
        print(f"\nbench: {len(results)} cells")
        for (name, b, d), libs in sorted(by_cell.items()):
            am = libs.get("am", {}).get("median_us")
            ours = libs.get("ours", {}).get("median_us")
            ta = libs.get("ta", {}).get("median_us")
            line = f"  {name:<24} b={b:<4} {d:<5}"
            if ours is not None:
                line += f" ours={ours:8.2f}us"
            if am is not None:
                line += f" am={am:10.2f}us"
                if ours:
                    line += f" speedup={am / ours:6.2f}x"
            if ta is not None:
                line += f" ta={ta:8.2f}us"
            print(line)
