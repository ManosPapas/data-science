"""Tests for analytics.experiment and analytics.causal."""

from __future__ import annotations

import numpy as np

from core.analytics import causal, experiment


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
