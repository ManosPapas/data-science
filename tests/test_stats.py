"""Tests for analytics.stats."""

from __future__ import annotations

import polars as pl

from core.analytics import stats


def test_summary_lists_every_column() -> None:
    out = stats.summary(pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))
    assert set(out["column"].to_list()) == {"a", "b"}


def test_welch_t_test_returns_pvalue() -> None:
    result = stats.welch_t_test([1, 2, 3, 4], [3, 4, 5, 6])
    assert 0.0 <= result.p_value <= 1.0


def test_compare_groups_picks_a_test() -> None:
    df = pl.DataFrame({"v": [1.0, 2, 3, 10, 11, 12], "g": ["a", "a", "a", "b", "b", "b"]})
    out = stats.compare_groups(df, value="v", group="g")
    assert out["test"] in {"welch_t", "mann_whitney"}
    assert "effect_size" in out
