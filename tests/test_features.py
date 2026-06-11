"""Tests for the stateless feature transforms."""

from __future__ import annotations

import polars as pl

from core.features import clean, geo, period, temporal, text, transform, validate


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


def test_transform_aggregate() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
    out = transform.aggregate(df, "g", {"v": "sum"})
    assert out.filter(pl.col("g") == "a")["v_sum"].item() == 3.0


def test_transform_join_validates_cardinality() -> None:
    left = pl.DataFrame({"k": [1, 2, 2], "x": [1, 2, 3]})
    right = pl.DataFrame({"k": [1, 2], "y": ["a", "b"]})
    out = transform.join(left, right, on="k", how="left", validate="m:1")
    assert out.height == 3
    assert "y" in out.columns


def test_temporal_add_calendar() -> None:
    from datetime import date

    out = temporal.add_calendar(pl.DataFrame({"d": [date(2024, 3, 15)]}), "d")
    assert out["year"].item() == 2024
    assert out["month"].item() == 3


def test_period_window_slices_recent() -> None:
    from datetime import date

    df = pl.DataFrame({"d": [date(2024, 1, 1), date(2024, 6, 1)], "v": [1.0, 2.0]})
    recent = period.window(df, "d", "30d", reference=date(2024, 6, 15))
    assert recent.height == 1


def test_period_year_over_year() -> None:
    from datetime import date

    df = pl.DataFrame({"d": [date(2023, 5, 1), date(2024, 5, 1)], "v": [100.0, 130.0]})
    result = period.year_over_year(df, "d", "v", period="month", reference=date(2024, 5, 15))
    assert result.current == 130.0
    assert result.previous == 100.0


def test_text_normalize() -> None:
    out = text.normalize(pl.DataFrame({"t": ["  Hello   World "]}), "t")
    assert out["t"].item() == "hello world"


def test_geo_haversine() -> None:
    same = geo.haversine([51.5], [-0.13], [51.5], [-0.13])
    assert abs(float(same[0])) < 1e-6
    apart = geo.haversine([51.5], [-0.13], [48.85], [2.35])
    assert float(apart[0]) > 0.0


def test_validate_check_schema_reports_missing() -> None:
    problems = validate.check_schema(
        pl.DataFrame({"a": [1, 2, 3]}), required=["a", "missing"], raise_on_error=False
    )
    assert any("missing" in p for p in problems)
