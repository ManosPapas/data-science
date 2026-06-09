"""Temporal feature engineering — calendar parts, lags, rolling windows, holidays.

Stateless ``f(df, ...) -> pl.DataFrame``. For lags/rolling, sort the frame by time first (and pass
``by`` to keep windows within a group).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import polars as pl


def add_calendar(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Add year / quarter / month / week / weekday / day / is_weekend / is_month_end columns."""
    d = pl.col(date_col)
    return df.with_columns(
        d.dt.year().alias("year"),
        d.dt.quarter().alias("quarter"),
        d.dt.month().alias("month"),
        d.dt.week().alias("week"),
        d.dt.weekday().alias("weekday"),
        d.dt.day().alias("day"),
        (d.dt.weekday() >= 6).alias("is_weekend"),
        (d == d.dt.month_end()).alias("is_month_end"),
    )


def add_holiday_flags(df: pl.DataFrame, date_col: str, *, countries: Sequence[str]) -> pl.DataFrame:
    """Add an ``is_holiday_<country>`` boolean per country (uses the ``holidays`` library)."""
    import holidays

    years = df.select(pl.col(date_col).dt.year().unique()).to_series().to_list()
    exprs = []
    for country in countries:
        calendar = holidays.country_holidays(country, years=years)
        holiday_dates = list(calendar.keys())
        exprs.append(pl.col(date_col).is_in(holiday_dates).alias(f"is_holiday_{country}"))
    return df.with_columns(exprs)


def add_lags(
    df: pl.DataFrame, column: str, *, lags: Sequence[int], by: str | Sequence[str] | None = None
) -> pl.DataFrame:
    """Add ``<column>_lag<k>`` for each k (shifted within ``by`` groups if given)."""
    exprs = []
    for lag in lags:
        expr = pl.col(column).shift(lag)
        if by is not None:
            expr = pl.col(column).shift(lag).over(by)
        exprs.append(expr.alias(f"{column}_lag{lag}"))
    return df.with_columns(exprs)


def add_rolling(
    df: pl.DataFrame,
    column: str,
    *,
    windows: Sequence[int],
    stat: str = "mean",
    by: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    """Add rolling-window features (mean/sum/std/min/max) for each window size."""
    exprs = []
    for window in windows:
        expr = getattr(pl.col(column), f"rolling_{stat}")(window_size=window)
        if by is not None:
            expr = getattr(pl.col(column), f"rolling_{stat}")(window_size=window).over(by)
        exprs.append(expr.alias(f"{column}_roll{window}_{stat}"))
    return df.with_columns(exprs)


def cyclical_encode(df: pl.DataFrame, column: str, *, period: int) -> pl.DataFrame:
    """Encode a cyclical integer column (month, hour, ...) as sin/cos so 12 is near 1."""
    radians = pl.col(column) / period * 2 * math.pi
    return df.with_columns(
        radians.sin().alias(f"{column}_sin"), radians.cos().alias(f"{column}_cos")
    )


def date_diff(
    df: pl.DataFrame, *, start: str, end: str, unit: str = "days", output: str = "date_diff"
) -> pl.DataFrame:
    """Add the difference ``end - start`` in ``unit`` (days/hours/minutes/seconds)."""
    delta = pl.col(end) - pl.col(start)
    return df.with_columns(getattr(delta.dt, f"total_{unit}")().alias(output))


def age_from(df: pl.DataFrame, dob_col: str, *, output: str = "age") -> pl.DataFrame:
    """Add an approximate age in whole years from a date-of-birth column (relative to today)."""
    days = (pl.lit(date.today()) - pl.col(dob_col)).dt.total_days()
    return df.with_columns((days / 365.25).floor().cast(pl.Int32).alias(output))


def to_period(
    df: pl.DataFrame, date_col: str, *, every: str = "1mo", output: str | None = None
) -> pl.DataFrame:
    """Truncate a date column to a period start (e.g. '1d', '1w', '1mo') for grouping."""
    return df.with_columns(
        pl.col(date_col).dt.truncate(every).alias(output or f"{date_col}_period")
    )
