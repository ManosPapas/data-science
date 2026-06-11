"""Cleaning & type-parsing transforms — raw frames into trustworthy, memory-lean ones.

Every function is stateless (``f(df, ...) -> pl.DataFrame``) and composes via ``df.pipe(...)``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import polars as pl
import polars.selectors as cs


def _snake(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", cleaned) or "column"


def standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename every column to clean snake_case."""
    return df.rename({col: _snake(col) for col in df.columns})


def cast_schema(df: pl.DataFrame, schema: dict[str, Any]) -> pl.DataFrame:
    """Cast columns to the given Polars dtypes (strict — raises on a bad value)."""
    return df.cast(cast(Any, schema))


def coerce(df: pl.DataFrame, schema: dict[str, Any]) -> pl.DataFrame:
    """Cast columns to the given dtypes, turning values that don't fit into null (non-strict)."""
    return df.cast(cast(Any, schema), strict=False)


def _string_dtype(series: pl.Series) -> Any | None:
    non_null = series.drop_nulls()
    if non_null.is_empty():
        return None
    if non_null.cast(pl.Int64, strict=False).null_count() == 0:
        return pl.Int64
    if non_null.cast(pl.Float64, strict=False).null_count() == 0:
        return pl.Float64
    try:
        if non_null.str.to_date(strict=False).null_count() == 0:
            return pl.Date
    except pl.exceptions.ComputeError:
        pass  # no inferable date format — not a date column, fall through
    if non_null.n_unique() <= max(20, int(0.05 * non_null.len())):
        return pl.Categorical
    return None


def infer_schema(df: pl.DataFrame) -> dict[str, Any]:
    """Suggest a better dtype for each string column based on its content (best-effort)."""
    suggestions: dict[str, Any] = {}
    for name, dtype in df.schema.items():
        if dtype == pl.Utf8:
            inferred = _string_dtype(df[name])
            if inferred is not None:
                suggestions[name] = inferred
    return suggestions


def auto_cast(df: pl.DataFrame) -> pl.DataFrame:
    """Parse string columns into inferred dtypes: numbers, ISO dates, low-card categoricals."""
    schema = infer_schema(df)
    date_cols = [name for name, dtype in schema.items() if dtype == pl.Date]
    other = {name: dtype for name, dtype in schema.items() if dtype != pl.Date}
    out = df.cast(cast(Any, other), strict=False) if other else df
    if date_cols:
        out = out.with_columns([pl.col(name).str.to_date(strict=False) for name in date_cols])
    return out


def downcast(df: pl.DataFrame) -> pl.DataFrame:
    """Shrink numeric dtypes and convert low-cardinality strings to Categorical (saves RAM)."""
    ints = [
        pl.col(name).shrink_dtype()
        for name, dtype in df.schema.items()
        if dtype in (pl.Int64, pl.Int32, pl.Int16, pl.UInt64, pl.UInt32)
    ]
    floats = [
        pl.col(name).cast(pl.Float32) for name, dtype in df.schema.items() if dtype == pl.Float64
    ]
    out = df.with_columns([*ints, *floats]) if ints or floats else df
    height = out.height
    categoricals = [
        pl.col(name).cast(pl.Categorical)
        for name, dtype in out.schema.items()
        if dtype == pl.Utf8 and height and out[name].n_unique() / height < 0.5
    ]
    return out.with_columns(categoricals) if categoricals else out


def fill_missing(
    df: pl.DataFrame,
    *,
    strategy: str = "forward",
    value: Any = None,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Fill nulls: constant / forward / backward / mean / min / max / zero, or median / mode."""
    cols = list(columns) if columns is not None else df.columns
    if strategy == "constant":
        return df.with_columns([pl.col(col).fill_null(value) for col in cols])
    if strategy == "median":
        return df.with_columns([pl.col(col).fill_null(pl.col(col).median()) for col in cols])
    if strategy == "mode":
        return df.with_columns([pl.col(col).fill_null(pl.col(col).mode().first()) for col in cols])
    return df.with_columns([pl.col(col).fill_null(strategy=cast(Any, strategy)) for col in cols])


def drop_duplicate_rows(
    df: pl.DataFrame, *, subset: Sequence[str] | None = None, keep: str = "first"
) -> pl.DataFrame:
    """Drop duplicate rows (optionally on a subset of columns), preserving order."""
    return df.unique(subset=subset, keep=cast(Any, keep), maintain_order=True)


def drop_duplicate_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns whose name repeats (keep the first occurrence)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for col in df.columns:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return df.select(ordered)


def winsorize(
    df: pl.DataFrame, columns: Sequence[str], *, lower: float = 0.01, upper: float = 0.99
) -> pl.DataFrame:
    """Clip the given numeric columns to their lower/upper quantiles."""
    exprs = [pl.col(col).clip(df[col].quantile(lower), df[col].quantile(upper)) for col in columns]
    return df.with_columns(exprs)


def filter_range(df: pl.DataFrame, bounds: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Keep only rows where each column falls within its (min, max)."""
    out = df
    for col, (lo, hi) in bounds.items():
        out = out.filter((pl.col(col) >= lo) & (pl.col(col) <= hi))
    return out


def clean_text(
    df: pl.DataFrame, columns: Sequence[str], *, lower: bool = False, strip: bool = True
) -> pl.DataFrame:
    """Normalize string columns (strip whitespace, optionally lowercase)."""
    exprs = []
    for col in columns:
        expr = pl.col(col)
        if strip:
            expr = expr.str.strip_chars()
        if lower:
            expr = expr.str.to_lowercase()
        exprs.append(expr.alias(col))
    return df.with_columns(exprs)


def extract_digits(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Replace a string column with only its digit characters."""
    return df.with_columns(pl.col(column).str.replace_all(r"\D", "").alias(column))


def drop_constant(df: pl.DataFrame) -> pl.DataFrame:
    """Drop zero-variance columns (a single distinct value, including all-null)."""
    return df.select([col for col in df.columns if df[col].n_unique() > 1])


def drop_highly_correlated(df: pl.DataFrame, *, threshold: float = 0.95) -> pl.DataFrame:
    """Drop later numeric columns that correlate with an earlier one above ``threshold``."""
    numeric = df.select(cs.numeric())
    names = numeric.columns
    if len(names) < 2:
        return df
    matrix = np.corrcoef(numeric.drop_nulls().to_numpy(), rowvar=False)
    drop: set[str] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[j] not in drop and abs(matrix[i, j]) >= threshold:
                drop.add(names[j])
    return df.drop(drop)
