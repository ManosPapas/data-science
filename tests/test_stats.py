"""Tests for analytics.stats."""

from __future__ import annotations

import numpy as np
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


def test_normality_test_ks_method(rng: np.random.Generator) -> None:
    assert stats.normality_test(rng.normal(0.0, 1.0, 300), method="ks").p_value > 0.05
    assert stats.normality_test(rng.exponential(1.0, 300), method="ks").p_value < 0.01


def test_fit_distribution_recovers_normal_params(rng: np.random.Generator) -> None:
    fit = stats.fit_distribution(rng.normal(10.0, 2.0, 2000), "norm")
    assert abs(fit["params"][0] - 10.0) < 0.2  # loc = mean
    assert abs(fit["params"][1] - 2.0) < 0.2  # scale = std
    assert fit["ks_p"] > 0.05  # the fitted shape matches the sample


def test_best_distribution_prefers_the_true_family(rng: np.random.Generator) -> None:
    ranked = stats.best_distribution(rng.exponential(2.0, 1500), candidates=("norm", "expon"))
    assert ranked["dist"][0] == "expon"  # lower AIC first


def test_missingness_dependence_separates_mar_from_mcar(rng: np.random.Generator) -> None:
    df = pl.DataFrame(
        {
            "driver": rng.normal(0.0, 1.0, 400),
            "noise": rng.normal(0.0, 1.0, 400),
            "v": rng.normal(5.0, 1.0, 400),
        }
    ).with_columns(pl.when(pl.col("driver") > 0.3).then(None).otherwise(pl.col("v")).alias("v"))
    out = stats.missingness_dependence(df, "v")
    assert out.filter(pl.col("column") == "driver")["p_value"][0] < 0.001  # MAR on driver
    assert out.filter(pl.col("column") == "noise")["p_value"][0] > 0.05  # unrelated
