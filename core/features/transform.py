"""Reshape, aggregate, and math transforms — stateless ``f(df, ...) -> pl.DataFrame``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def aggregate(df: pl.DataFrame, by: str | Sequence[str], metrics: dict[str, str]) -> pl.DataFrame:
    """Group by ``by`` and aggregate: ``metrics`` maps column -> agg ('sum'/'mean'/'count'/...)."""
    keys = [by] if isinstance(by, str) else list(by)
    exprs = [getattr(pl.col(col), how)().alias(f"{col}_{how}") for col, how in metrics.items()]
    return df.group_by(keys).agg(exprs)


def pivot(
    df: pl.DataFrame, *, on: str, index: str | Sequence[str], values: str, agg: str = "first"
) -> pl.DataFrame:
    """Long -> wide: spread distinct ``on`` values into columns of ``values``."""
    return df.pivot(on, index=index, values=values, aggregate_function=agg)


def unpivot(df: pl.DataFrame, *, on: Sequence[str], index: str | Sequence[str]) -> pl.DataFrame:
    """Wide -> long: melt the ``on`` columns into variable/value pairs."""
    return df.unpivot(on=on, index=index)


def join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: str | Sequence[str],
    how: str = "inner",
    validate: str = "m:m",
) -> pl.DataFrame:
    """Join two frames with a cardinality check (e.g. ``validate='1:m'``)."""
    return left.join(right, on=on, how=how, validate=validate)


def discretize(
    df: pl.DataFrame,
    column: str,
    *,
    breaks: Sequence[float] | None = None,
    quantiles: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Bin a numeric column by explicit ``breaks`` or by ``quantiles`` into ``<column>_bin``."""
    expr = pl.col(column)
    if quantiles is not None:
        binned = expr.qcut(quantiles, labels=labels)
    elif breaks is not None:
        binned = expr.cut(breaks, labels=labels)
    else:
        raise ValueError("provide either breaks or quantiles")
    return df.with_columns(binned.alias(f"{column}_bin"))


def pct_of_total(
    df: pl.DataFrame, column: str, *, over: str | Sequence[str] | None = None
) -> pl.DataFrame:
    """Add ``<column>_pct``: each value as a % of the total (optionally within ``over``)."""
    total = pl.col(column).sum().over(over) if over is not None else pl.col(column).sum()
    return df.with_columns((pl.col(column) / total * 100).alias(f"{column}_pct"))


def resample(df: pl.DataFrame, *, time_col: str, every: str, aggs: Sequence[Any]) -> pl.DataFrame:
    """Downsample a time series: bucket ``time_col`` by ``every`` (e.g. '1mo') and aggregate."""
    return df.sort(time_col).group_by_dynamic(time_col, every=every).agg(aggs)


def log1p(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Apply ``log(1 + x)`` to the given columns in place."""
    return df.with_columns([pl.col(col).log1p().alias(col) for col in columns])


def clip(
    df: pl.DataFrame, column: str, *, lower: float | None = None, upper: float | None = None
) -> pl.DataFrame:
    """Clamp a column to ``[lower, upper]``."""
    return df.with_columns(pl.col(column).clip(lower, upper).alias(column))


def rank(
    df: pl.DataFrame, column: str, *, method: str = "average", descending: bool = False
) -> pl.DataFrame:
    """Add ``<column>_rank`` ranking the column."""
    return df.with_columns(
        pl.col(column).rank(method=method, descending=descending).alias(f"{column}_rank")
    )


def sample(
    df: pl.DataFrame, *, n: int | None = None, fraction: float | None = None, seed: int = 42
) -> pl.DataFrame:
    """Random subset by row count ``n`` or ``fraction``."""
    if n is None and fraction is None:
        raise ValueError("provide n or fraction")
    return df.sample(n=n, fraction=fraction, seed=seed)


def stratified_sample(
    df: pl.DataFrame, by: str | Sequence[str], *, fraction: float, seed: int = 42
) -> pl.DataFrame:
    """Sample ``fraction`` of rows within each ``by`` group (preserves group proportions)."""
    rank_in_group = pl.int_range(pl.len()).shuffle(seed=seed).over(by)
    group_size = pl.len().over(by)
    return (
        df.with_columns(rank_in_group.alias("_r"), group_size.alias("_n"))
        .filter(pl.col("_r") < (pl.col("_n") * fraction).ceil())
        .drop("_r", "_n")
    )


def apply_rate(
    df: pl.DataFrame, *, value: str, rate: str, output: str | None = None
) -> pl.DataFrame:
    """Multiply a ``value`` column by a ``rate`` column (e.g. currency conversion)."""
    return df.with_columns((pl.col(value) * pl.col(rate)).alias(output or f"{value}_converted"))


def explode_json(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Unnest a struct column (decoding it first if it is a JSON string) into top-level columns."""
    if df[column].dtype == pl.Utf8:
        df = df.with_columns(pl.col(column).str.json_decode())
    return df.unnest(column)
