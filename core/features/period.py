"""Time-period slicing and period-over-period comparison (WoW / MoM / QoQ / YoY).

Slice a frame to rolling day-windows (7/14/28/30/60/90/120/365) or calendar periods (week/month/
quarter/year), then compare a metric across consecutive periods, an explicit pair, or year-on-year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import polars as pl

WINDOW_DAYS = {
    "7d": 7,
    "14d": 14,
    "28d": 28,
    "30d": 30,
    "60d": 60,
    "90d": 90,
    "120d": 120,
    "365d": 365,
}
PERIODS = ("day", "week", "month", "quarter", "year")


@dataclass(frozen=True)
class Comparison:
    """A current-vs-previous metric comparison with the absolute and relative change."""

    current: float
    previous: float
    change: float
    pct_change: float | None
    label: str


def _reference_date(df: pl.DataFrame, date_col: str, reference: date | None) -> date:
    if reference is not None:
        return reference
    latest: date = df[date_col].max()
    return latest


def _shift_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def _period_bounds(reference: date, period: str, *, offset: int = 0) -> tuple[date, date]:
    """Start/end of the ``period`` around ``reference``, shifted back ``offset`` periods."""
    if period == "day":
        day = reference - timedelta(days=offset)
        return day, day
    if period == "week":
        monday = reference - timedelta(days=reference.weekday()) - timedelta(weeks=offset)
        return monday, monday + timedelta(days=6)
    if period == "month":
        start = _shift_months(reference.replace(day=1), -offset)
        return start, _shift_months(start, 1) - timedelta(days=1)
    if period == "quarter":
        quarter_start_month = (reference.month - 1) // 3 * 3 + 1
        start = _shift_months(date(reference.year, quarter_start_month, 1), -offset * 3)
        return start, _shift_months(start, 3) - timedelta(days=1)
    if period == "year":
        return date(reference.year - offset, 1, 1), date(reference.year - offset, 12, 31)
    raise ValueError(f"unknown period: {period}")


def _aggregate(frame: pl.DataFrame, value: str, agg: str) -> float:
    series = frame[value]
    if not series.len():
        return 0.0
    result: float = getattr(series, agg)()
    return float(result)


# --- Slicing ------------------------------------------------------------------------------------


def between(df: pl.DataFrame, date_col: str, start: date, end: date) -> pl.DataFrame:
    """Rows with ``date_col`` in [start, end] (inclusive)."""
    return df.filter(pl.col(date_col).is_between(start, end))


def last_n_days(
    df: pl.DataFrame, date_col: str, n: int, *, reference: date | None = None
) -> pl.DataFrame:
    """Rows within the last ``n`` days up to ``reference`` (defaults to the latest date)."""
    end = _reference_date(df, date_col, reference)
    return between(df, date_col, end - timedelta(days=n - 1), end)


def window(
    df: pl.DataFrame, date_col: str, key: str = "30d", *, reference: date | None = None
) -> pl.DataFrame:
    """Slice by a named day-window from WINDOW_DAYS ('7d', '30d', '90d', ...)."""
    return last_n_days(df, date_col, WINDOW_DAYS[key], reference=reference)


def this_period(
    df: pl.DataFrame, date_col: str, period: str = "month", *, reference: date | None = None
) -> pl.DataFrame:
    """Rows in the calendar ``period`` (week/month/quarter/year) containing the reference date."""
    start, end = _period_bounds(_reference_date(df, date_col, reference), period)
    return between(df, date_col, start, end)


def previous_period(
    df: pl.DataFrame, date_col: str, period: str = "month", *, reference: date | None = None
) -> pl.DataFrame:
    """Rows in the period immediately before the reference period."""
    start, end = _period_bounds(_reference_date(df, date_col, reference), period, offset=1)
    return between(df, date_col, start, end)


# --- Comparison ---------------------------------------------------------------------------------


def compare_windows(
    df: pl.DataFrame,
    date_col: str,
    value: str,
    window_a: tuple[date, date],
    window_b: tuple[date, date],
    *,
    agg: str = "sum",
    labels: tuple[str, str] = ("a", "b"),
) -> Comparison:
    """Compare an aggregate of ``value`` across two explicit windows (e.g. Q1 vs Q2)."""
    a = _aggregate(between(df, date_col, *window_a), value, agg)
    b = _aggregate(between(df, date_col, *window_b), value, agg)
    change = a - b
    return Comparison(a, b, change, change / b if b else None, f"{labels[0]} vs {labels[1]}")


def compare_periods(
    df: pl.DataFrame,
    date_col: str,
    value: str,
    *,
    period: str = "month",
    agg: str = "sum",
    reference: date | None = None,
) -> Comparison:
    """Compare ``value`` for the latest ``period`` vs the one before it (WoW / MoM / QoQ)."""
    ref = _reference_date(df, date_col, reference)
    current = _period_bounds(ref, period, offset=0)
    previous = _period_bounds(ref, period, offset=1)
    return compare_windows(
        df, date_col, value, current, previous, agg=agg, labels=(f"{period}", f"prev {period}")
    )


def year_over_year(
    df: pl.DataFrame,
    date_col: str,
    value: str,
    *,
    period: str = "month",
    agg: str = "sum",
    reference: date | None = None,
) -> Comparison:
    """Compare the current ``period`` to the same period one year earlier."""
    ref = _reference_date(df, date_col, reference)
    prior_ref = date(ref.year - 1, ref.month, min(ref.day, 28))
    current = _period_bounds(ref, period, offset=0)
    prior = _period_bounds(prior_ref, period, offset=0)
    return compare_windows(
        df, date_col, value, current, prior, agg=agg, labels=(period, "year ago")
    )


def week_over_week(df: pl.DataFrame, date_col: str, value: str, **kwargs: Any) -> Comparison:
    """WoW shortcut for :func:`compare_periods` with ``period='week'``."""
    return compare_periods(df, date_col, value, period="week", **kwargs)


def month_over_month(df: pl.DataFrame, date_col: str, value: str, **kwargs: Any) -> Comparison:
    """MoM shortcut for :func:`compare_periods` with ``period='month'``."""
    return compare_periods(df, date_col, value, period="month", **kwargs)


def quarter_over_quarter(df: pl.DataFrame, date_col: str, value: str, **kwargs: Any) -> Comparison:
    """QoQ shortcut for :func:`compare_periods` with ``period='quarter'``."""
    return compare_periods(df, date_col, value, period="quarter", **kwargs)


def compare_periods_by(
    df: pl.DataFrame,
    date_col: str,
    value: str,
    group: str,
    *,
    period: str = "month",
    agg: str = "sum",
    reference: date | None = None,
) -> pl.DataFrame:
    """Per-group current-vs-previous-period aggregate + change (e.g. revenue by region, MoM)."""
    ref = _reference_date(df, date_col, reference)
    current = between(df, date_col, *_period_bounds(ref, period, offset=0))
    previous = between(df, date_col, *_period_bounds(ref, period, offset=1))
    current_agg = current.group_by(group).agg(getattr(pl.col(value), agg)().alias("current"))
    previous_agg = previous.group_by(group).agg(getattr(pl.col(value), agg)().alias("previous"))
    joined = current_agg.join(previous_agg, on=group, how="full", coalesce=True).fill_null(0)
    return joined.with_columns(
        (pl.col("current") - pl.col("previous")).alias("change"),
        pl.when(pl.col("previous") != 0)
        .then((pl.col("current") - pl.col("previous")) / pl.col("previous"))
        .otherwise(None)
        .alias("pct_change"),
    ).sort(group)
