"""Splitting and cross-validation strategies — random, stratified, group, and time-based."""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.model_selection import (
    train_test_split as _sk_split,
)


def train_test_split(
    df: pl.DataFrame, *, test_size: float = 0.2, stratify: str | None = None, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random (optionally stratified) split into (train, test) frames."""
    strata = df[stratify].to_numpy() if stratify is not None else None
    train_idx, test_idx = _sk_split(
        list(range(df.height)), test_size=test_size, random_state=seed, stratify=strata
    )
    return df[train_idx], df[test_idx]


def train_val_test_split(
    df: pl.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: str | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Three-way split into (train, val, test); stratified on ``stratify`` if given."""
    strata = df[stratify].to_numpy() if stratify is not None else None
    train_val_idx, test_idx = _sk_split(
        list(range(df.height)), test_size=test_size, random_state=seed, stratify=strata
    )
    inner_strata = df[stratify].to_numpy()[train_val_idx] if stratify is not None else None
    train_idx, val_idx = _sk_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=inner_strata,
    )
    return df[train_idx], df[val_idx], df[test_idx]


def group_split(
    df: pl.DataFrame, group: str, *, test_size: float = 0.2, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split so every value of ``group`` stays wholly in train or test (no entity leakage)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(list(range(df.height)), groups=df[group].to_numpy()))
    return df[train_idx.tolist()], df[test_idx.tolist()]


def time_split(
    df: pl.DataFrame, time_col: str, *, test_size: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological split: earliest rows train, latest ``test_size`` fraction test (no shuffle)."""
    ordered = df.sort(time_col)
    cut = int(ordered.height * (1 - test_size))
    return ordered[:cut], ordered[cut:]


def make_cv(
    strategy: str = "kfold",
    *,
    n_splits: int = 5,
    n_repeats: int = 2,
    shuffle: bool = True,
    seed: int = 42,
) -> Any:
    """Build a cross-validation splitter: kfold / stratified / group / repeated / timeseries."""
    state = seed if shuffle else None
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=state)
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=state)
    if strategy == "group":
        return GroupKFold(n_splits=n_splits)
    if strategy == "repeated":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"unknown cv strategy: {strategy}")
