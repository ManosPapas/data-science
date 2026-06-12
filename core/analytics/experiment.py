"""A/B experiment analysis — lift, significance, confidence interval, and a verdict.

Builds on the primitives in ``analytics.stats`` (Welch t-test, two-proportion z-test). Also ships
the experiment-quality toolkit: ``srm_check`` (is the split broken?), ``cuped_adjust`` (variance
reduction from a pre-experiment covariate), and ``msprt_means`` (an always-valid p-value that is
safe to peek at while the experiment runs). The ``bayes_*`` pair gives the Bayesian read of the
same data — P(treatment better) and expected loss instead of a p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
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


@dataclass(frozen=True)
class BayesResult:
    """Posterior summary of treatment vs control."""

    control: float
    treatment: float
    absolute_effect: float
    lift: float
    prob_treatment_better: float
    expected_loss: float
    credible_interval: tuple[float, float]


def _bayes_summary(
    control_draws: NDArray[np.float64], treatment_draws: NDArray[np.float64], alpha: float
) -> BayesResult:
    effect = treatment_draws - control_draws
    control_mean = float(control_draws.mean())
    lower, upper = np.quantile(effect, [alpha / 2, 1 - alpha / 2])
    return BayesResult(
        control_mean,
        float(treatment_draws.mean()),
        float(effect.mean()),
        float(effect.mean() / control_mean) if control_mean else float("nan"),
        float((effect > 0).mean()),
        float(np.maximum(-effect, 0.0).mean()),
        (float(lower), float(upper)),
    )


def bayes_conversions(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.05,
    draws: int = 100_000,
    seed: int = 42,
) -> BayesResult:
    """Bayesian A/B test for conversion rates (Beta-Binomial posterior, Monte Carlo).

    Instead of a p-value you get ``prob_treatment_better`` = P(treatment rate > control rate | data)
    and ``expected_loss`` = the conversion-rate you expect to give up if you ship treatment and it
    is actually worse — ship when that loss is below your tolerance. ``prior`` is the Beta(a, b)
    prior (default uniform); use it to encode history when data are thin, and read
    ``credible_interval`` as "the effect is in this range with 95% probability".
    """
    rng = np.random.default_rng(seed)
    a, b = prior
    control_draws = rng.beta(a + control_conversions, b + control_n - control_conversions, draws)
    treatment_draws = rng.beta(
        a + treatment_conversions, b + treatment_n - treatment_conversions, draws
    )
    return _bayes_summary(control_draws, treatment_draws, alpha)


def bayes_means(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    alpha: float = 0.05,
    draws: int = 100_000,
    seed: int = 42,
) -> BayesResult:
    """Bayesian comparison of a continuous metric via each mean's large-sample normal posterior.

    Approximates the posterior of each arm's mean as Normal(sample mean, sem) — fine from a few
    dozen observations per arm; for tiny samples or heavy tails prefer ``analyze_means`` or model
    the data explicitly. Same decision quantities as :func:`bayes_conversions`.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    control_draws = rng.normal(c.mean(), c.std(ddof=1) / np.sqrt(c.size), draws)
    treatment_draws = rng.normal(t.mean(), t.std(ddof=1) / np.sqrt(t.size), draws)
    return _bayes_summary(control_draws, treatment_draws, alpha)


def srm_check(
    counts: Sequence[int], *, expected: Sequence[float] | None = None
) -> stats.TestResult:
    """Sample-ratio-mismatch chi-square: do the arm sizes match the intended split?

    A small p-value (< 0.001 is the usual alarm) means assignment is broken — fix the experiment
    before reading any metric. ``expected`` are split ratios (default: equal arms).
    """
    from scipy.stats import chisquare

    observed = np.asarray(counts, dtype=float)
    ratios = (
        np.asarray(expected, dtype=float)
        if expected is not None
        else np.full(observed.size, 1.0 / observed.size)
    )
    ratios = ratios / ratios.sum()
    result = chisquare(observed, f_exp=observed.sum() * ratios)
    return stats.TestResult(float(result.statistic), float(result.pvalue))


def cuped_adjust(
    metric: ArrayLike, covariate: ArrayLike, *, theta: float | None = None
) -> NDArray[np.float64]:
    """CUPED variance reduction: residualize ``metric`` on a pre-experiment ``covariate``.

    Returns ``metric - theta * (covariate - mean(covariate))`` — same mean, lower variance, so the
    same effect is detectable with fewer users. Compute ``theta`` once on ALL units (both arms
    pooled) and pass it in; adjust each arm with that shared theta, then run ``analyze_means``.
    """
    m = np.asarray(metric, dtype=float)
    c = np.asarray(covariate, dtype=float)
    factor = float(np.cov(m, c)[0, 1] / c.var(ddof=1)) if theta is None else theta
    return np.asarray(m - factor * (c - c.mean()), dtype=float)


def msprt_means(control: ArrayLike, treatment: ArrayLike, *, tau: float | None = None) -> float:
    """Always-valid p-value (mixture SPRT) for a difference in means — safe to peek anytime.

    A fixed-horizon t-test is only valid when you look once, at the planned sample size; this
    p-value stays valid under continuous monitoring, so you may stop the moment it crosses alpha.
    ``tau`` is the prior scale of plausible effects (default: half the pooled standard deviation).
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    se2 = float(c.var(ddof=1) / c.size + t.var(ddof=1) / t.size)
    scale = tau if tau is not None else 0.5 * float(np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2))
    delta = float(t.mean() - c.mean())
    tau2 = scale * scale
    likelihood = float(
        np.sqrt(se2 / (se2 + tau2)) * np.exp(delta**2 * tau2 / (2 * se2 * (se2 + tau2)))
    )
    return min(1.0, 1.0 / likelihood)
