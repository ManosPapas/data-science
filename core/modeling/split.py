"""Train/test splitting — random, stratified, and time-based (for forecasting)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import polars as pl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split as _sk_split


def train_test_split(
    df: pl.DataFrame, *, test_size: float = 0.2, stratify: str | None = None, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random (optionally stratified) split into (train, test) frames."""
    strata = df[stratify].to_numpy() if stratify is not None else None
    train_idx, test_idx = _sk_split(
        list(range(df.height)), test_size=test_size, random_state=seed, stratify=strata
    )
    return df[train_idx], df[test_idx]


def time_split(
    df: pl.DataFrame, time_col: str, *, test_size: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological split: earliest rows train, latest ``test_size`` fraction test (no shuffle)."""
    ordered = df.sort(time_col)
    cut = int(ordered.height * (1 - test_size))
    return ordered[:cut], ordered[cut:]


def time_series_cv(n_samples: int, *, n_splits: int = 5) -> Iterator[tuple[Any, Any]]:
    """Yield (train_idx, test_idx) for expanding-window time-series cross-validation."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    yield from splitter.split(list(range(n_samples)))
