"""Tests for analytics.experiment, analytics.causal, and analytics.regression."""

from __future__ import annotations

import numpy as np
import polars as pl

from core.analytics import causal, experiment, regression


def test_analyze_means_detects_lift(rng: np.random.Generator) -> None:
    control = rng.normal(10.0, 2.0, 500)
    treatment = rng.normal(11.0, 2.0, 500)
    result = experiment.analyze_means(control, treatment)
    assert result.treatment > result.control
    assert result.verdict in {"win", "loss", "inconclusive"}


def test_analyze_conversions() -> None:
    result = experiment.analyze_conversions(100, 1000, 140, 1000)
    assert result.control == 0.1
    assert isinstance(result.significant, bool)


def test_srm_check_flags_a_broken_split() -> None:
    assert experiment.srm_check([5000, 5050]).p_value > 0.05
    assert experiment.srm_check([5000, 4000]).p_value < 0.001


def test_cuped_reduces_variance_and_keeps_the_mean(rng: np.random.Generator) -> None:
    covariate = rng.normal(50.0, 10.0, 2000)
    metric = covariate * 0.8 + rng.normal(0.0, 5.0, 2000)
    adjusted = experiment.cuped_adjust(metric, covariate)
    assert adjusted.var() < metric.var() * 0.6  # strong covariate -> big variance cut
    assert abs(adjusted.mean() - metric.mean()) < 1e-9  # the effect estimate is untouched


def test_msprt_means_is_calm_under_null_and_fires_on_effect(rng: np.random.Generator) -> None:
    control = rng.normal(10.0, 2.0, 500)
    assert experiment.msprt_means(control, rng.normal(10.0, 2.0, 500)) > 0.05
    assert experiment.msprt_means(control, rng.normal(12.0, 2.0, 500)) < 0.01


def test_difference_in_differences() -> None:
    assert abs(causal.difference_in_differences(0.8, 0.79, 0.8, 0.86) - 0.07) < 1e-9


def test_uplift() -> None:
    assert causal.uplift(np.ones(50), np.zeros(50)) == 1.0


def test_propensity_and_matching(rng: np.random.Generator) -> None:
    x = rng.normal(size=(200, 3))
    treatment = (x[:, 0] + rng.normal(0, 0.1, 200) > 0).astype(int)
    scores = causal.propensity_scores(x, treatment)
    assert scores.shape == (200,)
    matched = causal.match_on_propensity(scores, treatment, caliper=0.1)
    assert matched.ndim == 1


def test_bayes_conversions_decides_a_clear_winner() -> None:
    result = experiment.bayes_conversions(100, 1000, 150, 1000)
    assert result.prob_treatment_better > 0.99
    assert result.expected_loss < 0.005
    lower, upper = result.credible_interval
    assert lower < result.absolute_effect < upper


def test_bayes_conversions_stays_uncertain_when_arms_match() -> None:
    result = experiment.bayes_conversions(100, 1000, 101, 1000)
    assert 0.2 < result.prob_treatment_better < 0.8


def test_bayes_means_detects_a_lift(rng: np.random.Generator) -> None:
    result = experiment.bayes_means(rng.normal(10.0, 2.0, 400), rng.normal(10.5, 2.0, 400))
    assert result.prob_treatment_better > 0.95
    assert result.treatment > result.control


def test_ipw_ate_corrects_observed_confounding(rng: np.random.Generator) -> None:
    x = rng.normal(0.0, 1.0, 4000)  # confounder raises both uptake and outcome
    propensity = 1.0 / (1.0 + np.exp(-2.0 * x))
    treated = (rng.random(4000) < propensity).astype(int)
    y = 1.0 * treated + 2.0 * x + rng.normal(0.0, 0.5, 4000)
    naive = float(y[treated == 1].mean() - y[treated == 0].mean())
    adjusted = causal.ipw_ate(y, treated, propensity)
    assert abs(adjusted - 1.0) < abs(naive - 1.0)  # weighting removes most of the bias
    assert abs(adjusted - 1.0) < 0.3


def test_itt_tot_scales_by_compliance() -> None:
    assigned = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    treated = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # half of the assigned comply
    outcome = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = causal.itt_tot(assigned, treated, outcome)
    assert abs(result.itt - 1.0) < 1e-9
    assert abs(result.compliance - 0.5) < 1e-9
    assert abs(result.tot - 2.0) < 1e-9  # diluted ITT scaled back up


def test_iv_effect_beats_ols_under_confounding(rng: np.random.Generator) -> None:
    n = 5000
    instrument = rng.normal(0.0, 1.0, n)
    unobserved = rng.normal(0.0, 1.0, n)
    treatment = 0.8 * instrument + unobserved + rng.normal(0.0, 0.3, n)
    outcome = 1.5 * treatment + 2.0 * unobserved + rng.normal(0.0, 0.3, n)
    ols = float(np.cov(treatment, outcome)[0, 1] / np.cov(treatment, treatment)[0, 0])
    iv = causal.iv_effect(outcome, treatment, instrument)
    assert abs(iv - 1.5) < abs(ols - 1.5)
    assert abs(iv - 1.5) < 0.15


def test_subgroup_effects_finds_the_responsive_segment(rng: np.random.Generator) -> None:
    n = 600
    segment = np.array(["a", "b"] * (n // 2))
    treated = rng.integers(0, 2, n)
    lift = np.where(segment == "a", 3.0, 0.0) * treated
    df = pl.DataFrame({"seg": segment, "t": treated, "y": rng.normal(10.0, 1.0, n) + lift})
    out = causal.subgroup_effects(df, outcome="y", treatment="t", segment="seg")
    assert out.height == 2
    assert out["segment"][0] == "a"  # biggest effect first
    assert abs(out["effect"][0] - 3.0) < 0.5
    assert out["p_value"][0] < 0.001


def test_vif_flags_the_collinear_feature(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, 300)
    df = pl.DataFrame(
        {"a": a, "b": rng.normal(0.0, 1.0, 300), "dup": a * 0.97 + rng.normal(0.0, 0.05, 300)}
    )
    out = regression.vif(df)
    assert out["feature"][0] in {"a", "dup"}
    assert out["vif"][0] > 10.0
    assert out.filter(pl.col("feature") == "b")["vif"][0] < 2.0


def test_breusch_pagan_detects_heteroscedasticity(rng: np.random.Generator) -> None:
    x = rng.uniform(1.0, 10.0, 500)
    assert regression.breusch_pagan(rng.normal(0.0, 1.0, 500), x).p_value > 0.05
    assert regression.breusch_pagan(rng.normal(0.0, 1.0, 500) * x, x).p_value < 0.01


def test_durbin_watson_reads_autocorrelation(rng: np.random.Generator) -> None:
    assert 1.5 < regression.durbin_watson(rng.normal(0.0, 1.0, 500)) < 2.5
    assert regression.durbin_watson(np.cumsum(rng.normal(0.0, 1.0, 500))) < 1.0


def test_linear_assumptions_reports_every_check(rng: np.random.Generator) -> None:
    features = pl.DataFrame({"a": rng.normal(0.0, 1.0, 200), "b": rng.normal(0.0, 1.0, 200)})
    out = regression.linear_assumptions(features, rng.normal(0.0, 1.0, 200))
    assert set(out) == {
        "normality_p",
        "homoscedasticity_p",
        "durbin_watson",
        "max_vif",
        "max_vif_feature",
    }
    assert float(out["normality_p"]) > 0.05  # well-behaved residuals pass
