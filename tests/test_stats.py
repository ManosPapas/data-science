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


def test_bayes_rule_respects_the_base_rate() -> None:
    posterior = stats.bayes_rule(0.01, 0.99, 0.05)
    assert abs(posterior - 0.1667) < 0.01  # 99%-sensitive test, 1% prior -> only ~17%


def test_proportion_confidence_interval_brackets_the_rate() -> None:
    rate, lower, upper = stats.proportion_confidence_interval(30, 1000)
    assert lower < rate < upper
    assert lower > 0.0 and upper < 1.0


def test_bootstrap_ci_brackets_mean_and_median(rng: np.random.Generator) -> None:
    data = rng.normal(50.0, 5.0, 400)
    estimate, lower, upper = stats.bootstrap_ci(data, np.mean, n_resamples=500, seed=0)
    assert lower < estimate < upper
    assert lower < 50.0 < upper
    estimate, lower, upper = stats.bootstrap_ci(data, np.median, n_resamples=500, seed=0)
    assert lower < estimate < upper


def test_permutation_test_detects_shift_and_stays_calm(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, 60)
    shifted = stats.permutation_test(a, rng.normal(1.0, 1.0, 60), n_resamples=2000, seed=0)
    null = stats.permutation_test(a, rng.normal(0.0, 1.0, 60), n_resamples=2000, seed=0)
    assert shifted.p_value < 0.01
    assert null.p_value > 0.05


def test_simpsons_check_flags_a_reversal(rng: np.random.Generator) -> None:
    # within group y falls with x, but the high-x group sits higher overall
    x_a = rng.uniform(0.0, 1.0, 200)
    x_b = rng.uniform(2.0, 3.0, 200)
    y_a = -1.0 * x_a + rng.normal(0.0, 0.05, 200)
    y_b = -1.0 * x_b + 5.0 + rng.normal(0.0, 0.05, 200)
    df = pl.DataFrame(
        {
            "x": np.concatenate([x_a, x_b]),
            "y": np.concatenate([y_a, y_b]),
            "g": ["a"] * 200 + ["b"] * 200,
        }
    )
    out = stats.simpsons_check(df, x="x", y="y", group="g")
    assert out["reversal"] is True
    assert out["overall_slope"] > 0 > out["within_slope"]
    assert out["by_group"].height == 2


def test_entropy_and_information_gain() -> None:
    assert abs(stats.entropy(["a", "b", "a", "b"]) - 1.0) < 1e-9  # fair coin = 1 bit
    assert stats.entropy(["a", "a", "a"]) == 0.0
    df = pl.DataFrame({"f": ["x", "x", "y", "y"], "t": ["p", "p", "n", "n"]})
    assert abs(stats.information_gain(df, "f", "t") - 1.0) < 1e-9  # perfect predictor


def test_kl_divergence_zero_iff_identical() -> None:
    assert stats.kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert stats.kl_divergence([0.9, 0.1], [0.5, 0.5]) > 0.0


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


def test_fit_discrete_recovers_parameters(rng: np.random.Generator) -> None:
    from core.analytics import stats

    poisson_fit = stats.fit_discrete(rng.poisson(4.0, 4000), "poisson")
    assert abs(poisson_fit["params"]["mu"] - 4.0) < 0.15

    geometric_fit = stats.fit_discrete(rng.geometric(0.3, 4000), "geometric")
    assert abs(geometric_fit["params"]["p"] - 0.3) < 0.02

    nbinom_fit = stats.fit_discrete(rng.negative_binomial(5, 0.4, 4000), "nbinom")
    assert abs(nbinom_fit["params"]["n"] - 5.0) < 1.0
    assert abs(nbinom_fit["params"]["p"] - 0.4) < 0.05


def test_fit_discrete_validations(rng: np.random.Generator) -> None:
    import pytest

    from core.analytics import stats

    with pytest.raises(ValueError, match="non-negative integers"):
        stats.fit_discrete([1.5, 2.0], "poisson")
    with pytest.raises(ValueError, match="trials until success"):
        stats.fit_discrete([0, 1, 2], "geometric")
    with pytest.raises(ValueError, match="poisson"):
        stats.fit_discrete([1, 2], "zipf")


def test_best_discrete_prefers_nbinom_for_overdispersed(rng: np.random.Generator) -> None:
    from core.analytics import stats

    overdispersed = rng.negative_binomial(2, 0.2, 3000)  # variance >> mean
    ranked = stats.best_discrete(overdispersed)
    assert ranked["dist"][0] == "nbinom"
    assert "geometric" not in ranked["dist"].to_list()  # zeros knock geometric out

    poisson_data = rng.poisson(3.0, 3000)
    ranked_poisson = stats.best_discrete(poisson_data, candidates=("poisson", "nbinom"))
    # nbinom nests poisson, so AICs are close - poisson within 2.5 of the winner
    aic = dict(zip(ranked_poisson["dist"].to_list(), ranked_poisson["aic"].to_list(), strict=True))
    assert aic["poisson"] - min(aic.values()) < 2.5


def test_dispersion_check(rng: np.random.Generator) -> None:
    from core.analytics import stats

    poisson_counts = rng.poisson(4.0, 2000)
    calm = stats.dispersion_check(poisson_counts)
    assert not calm.overdispersed
    assert abs(calm.ratio - 1.0) < 0.15

    lumpy = rng.negative_binomial(2, 0.2, 2000)
    wild = stats.dispersion_check(lumpy)
    assert wild.overdispersed
    assert wild.ratio > 2.0
    assert wild.p_value < 0.001


def test_gamma_posterior_updates_correctly() -> None:
    import pytest

    from core.analytics import bayes

    posterior = bayes.gamma_posterior(30, 10.0)  # 30 events over 10 units of exposure
    assert posterior.mean == pytest.approx(31.0 / 11.0)
    assert posterior.lower < 3.0 < posterior.upper
    strong_prior = bayes.gamma_posterior(0, 1.0, prior=(50.0, 10.0))
    assert strong_prior.mean == pytest.approx(50.0 / 11.0)  # prior dominates thin data
    with pytest.raises(ValueError, match="positive"):
        bayes.gamma_posterior(5, 0.0)


def test_fit_discrete_binom_and_zip(rng: np.random.Generator) -> None:
    import pytest

    from core.analytics import stats

    binom_fit = stats.fit_discrete(rng.binomial(20, 0.35, 4000), "binom", trials=20)
    assert abs(binom_fit["params"]["p"] - 0.35) < 0.02
    with pytest.raises(ValueError, match="trials"):
        stats.fit_discrete([1, 2], "binom")

    structural_zero = rng.random(6000) < 0.3
    zip_data = np.where(structural_zero, 0, rng.poisson(5.0, 6000))
    zip_fit = stats.fit_discrete(zip_data, "zip")
    assert abs(zip_fit["params"]["pi"] - 0.3) < 0.04
    assert abs(zip_fit["params"]["mu"] - 5.0) < 0.25

    ranked = stats.best_discrete(zip_data)
    assert ranked["dist"][0] == "zip"  # the truth beats poisson/nbinom on AIC


def test_best_distribution_skips_incompatible(rng: np.random.Generator) -> None:
    from core.analytics import stats

    ranked = stats.best_distribution(rng.normal(10.0, 2.0, 500), candidates=("norm", "poisson"))
    assert ranked["dist"].to_list() == ["norm"]  # discrete name skipped, not fatal


def test_best_discrete_survives_underdispersed(rng: np.random.Generator) -> None:
    import pytest

    from core.analytics import stats

    # underdispersed counts have no interior nbinom MLE - must not crash
    constant_ish = np.full(500, 4) + rng.integers(0, 2, 500)  # variance << mean
    ranked = stats.best_discrete(constant_ish)
    assert ranked.height >= 1  # poisson/binom survive, nbinom is skipped cleanly
    with pytest.raises(ValueError, match="not overdispersed"):
        stats.fit_discrete(constant_ish, "nbinom")
