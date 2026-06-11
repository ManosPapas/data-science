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
