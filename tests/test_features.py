"""Tests for the stateless feature transforms."""

from __future__ import annotations

import polars as pl

from core.features import clean, period


def test_standardize_columns() -> None:
    out = clean.standardize_columns(pl.DataFrame({"A B": [1], "X-Y": [2]}))
    assert out.columns == ["a_b", "x_y"]


def test_fill_missing_constant() -> None:
    out = clean.fill_missing(pl.DataFrame({"a": [1, None, 3]}), strategy="constant", value=0)
    assert out["a"].null_count() == 0


def test_drop_duplicate_rows() -> None:
    assert clean.drop_duplicate_rows(pl.DataFrame({"x": [1, 1, 2]})).height == 2


def test_auto_cast_parses_dates_without_choking_on_text() -> None:
    df = pl.DataFrame({"d": ["2024-01-01", "2024-06-15"], "kind": ["x", "y"], "n": ["1", "2"]})
    out = clean.auto_cast(df)
    assert out["d"].dtype == pl.Date
    assert out["n"].dtype == pl.Int64
    assert out["kind"].dtype == pl.Categorical  # non-date text must not raise ComputeError


def test_month_over_month() -> None:
    from datetime import date

    df = pl.DataFrame(
        {
            "d": [date(2024, 1, 5), date(2024, 1, 20), date(2024, 2, 10), date(2024, 2, 25)],
            "v": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = period.month_over_month(df, "d", "v")
    assert result.current == 70.0
    assert result.previous == 30.0
