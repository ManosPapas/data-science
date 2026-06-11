"""Lightweight data-quality checks — fail fast before bad data reaches a model or a chart."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl


def check_schema(
    df: pl.DataFrame,
    *,
    required: Sequence[str] | None = None,
    non_null: Sequence[str] | None = None,
    unique: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate a frame; return problems found (raises if ``raise_on_error``)."""
    problems: list[str] = []
    columns = set(df.columns)

    for col in required or []:
        if col not in columns:
            problems.append(f"missing required column: {col}")

    for col in non_null or []:
        if col in columns and df[col].null_count() > 0:
            problems.append(f"nulls present in non-null column: {col}")

    for col in unique or []:
        if col in columns and df[col].n_unique() != df.height:
            problems.append(f"duplicate values in unique column: {col}")

    for col, (lo, hi) in (ranges or {}).items():
        if col not in columns:
            continue
        minimum: Any = df[col].min()
        maximum: Any = df[col].max()
        if minimum is not None and float(minimum) < lo:
            problems.append(f"{col} below minimum {lo} (saw {minimum})")
        if maximum is not None and float(maximum) > hi:
            problems.append(f"{col} above maximum {hi} (saw {maximum})")

    if problems and raise_on_error:
        raise ValueError("schema validation failed:\n  - " + "\n  - ".join(problems))
    return problems
