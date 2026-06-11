"""A/B experiment analysis — lift, significance, confidence interval, and a verdict.

Builds on the primitives in ``analytics.stats`` (Welch t-test, two-proportion z-test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from core.analytics import stats


@dataclass(frozen=True)
class ExperimentResult:
    """Outcome of comparing a metric between control and treatment."""

    control: float
    treatment: float
    absolute_effect: float
    lift: float
    p_value: float
    confidence_interval: tuple[float, float]
    significant: bool
    verdict: str


def _verdict(effect: float, significant: bool) -> str:
    if not significant:
        return "inconclusive"
    return "win" if effect > 0 else "loss"


def analyze_means(
    control: ArrayLike, treatment: ArrayLike, *, alpha: float = 0.05
) -> ExperimentResult:
    """Compare a continuous metric (e.g. revenue per user) between control and treatment."""
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    control_mean, treatment_mean = float(c.mean()), float(t.mean())
    effect = treatment_mean - control_mean
    test = stats.welch_t_test(c, t)
    se = float(np.sqrt(c.var(ddof=1) / c.size + t.var(ddof=1) / t.size))
    z = float(norm.ppf(1 - alpha / 2))
    significant = test.p_value < alpha
    lift = effect / control_mean if control_mean else float("nan")
    return ExperimentResult(
        control_mean,
        treatment_mean,
        effect,
        lift,
        test.p_value,
        (effect - z * se, effect + z * se),
        significant,
        _verdict(effect, significant),
    )


def analyze_conversions(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    *,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Compare a conversion rate between control and treatment (two-proportion z-test)."""
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    effect = treatment_rate - control_rate
    test = stats.proportions_test(
        [treatment_conversions, control_conversions], [treatment_n, control_n]
    )
    se = float(
        np.sqrt(
            control_rate * (1 - control_rate) / control_n
            + treatment_rate * (1 - treatment_rate) / treatment_n
        )
    )
    z = float(norm.ppf(1 - alpha / 2))
    significant = test.p_value < alpha
    lift = effect / control_rate if control_rate else float("nan")
    return ExperimentResult(
        control_rate,
        treatment_rate,
        effect,
        lift,
        test.p_value,
        (effect - z * se, effect + z * se),
        significant,
        _verdict(effect, significant),
    )
