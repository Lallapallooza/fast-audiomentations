"""Flat-list registry for the unified validate + bench harness.

One ``Row`` per ``(transform, library)`` cell. Rows are appended via ``add()``;
the runner iterates ``ROWS`` and dispatches to the right adapter / validator
by string field. Sweep dims (batch, dtype) are CLI flags, not registry data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

Lib = Literal["am", "ta", "ours"]
Validator = Literal["numeric", "statistical", "skip"]
Adapter = Literal["default", "with_lens"]


@dataclass(frozen=True)
class Row:
    """One (transform, library) cell.

    ``name`` groups rows across libraries for validation pairing. ``lib`` picks
    the call adapter. ``ctor`` is a zero-arg factory returning a fresh transform
    instance with ``p=1.0`` (always-apply); use closures to bake in deterministic
    parameter values for numeric validation. ``atol``/``rtol`` are used when
    ``validator == "numeric"``; ``moment_tol`` and ``n_trials`` apply to
    ``"statistical"``. ``adapter`` selects the call-shape adapter for transforms
    that need extra inputs (e.g. ``with_lens`` for AddBackgroundNoise).
    """

    name: str
    lib: Lib
    ctor: Callable[[], Any]
    atol: float = 1e-5
    rtol: float = 1e-5
    validator: Validator = "numeric"
    n_trials: int = 1
    moment_tol: float = 0.05
    adapter: Adapter = "default"
    note: str = ""


ROWS: list[Row] = []


def add(*rows: Row) -> None:
    """Append rows to the global registry. Order is preserved for reporting."""
    ROWS.extend(rows)
