"""Risk & uncertainty measures over outcome distributions — VaR, shortfall, drawdown, targets.

Everything here consumes a sample of outcomes (Monte Carlo draws from ``decision.simulate``,
bootstrap replicates, historical P&L periods) and prices its *bad tail*. Convention: outcomes are
values where bigger is better (profit, margin), so risk lives in the low quantiles.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def _outcomes(values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("need at least one outcome")
    return x


def value_at_risk(outcomes: ArrayLike, *, alpha: float = 0.05) -> float:
    """The alpha-quantile outcome: with confidence 1-alpha the result won't come in below this.

    VaR is a *threshold*, not an expectation — it says nothing about how bad the worst alpha of
    cases get (that is :func:`expected_shortfall`). Quoted on outcomes, so a VaR of -2M reads
    "5% chance of losing 2M or more".
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(np.quantile(_outcomes(outcomes), alpha))


def expected_shortfall(outcomes: ArrayLike, *, alpha: float = 0.05) -> float:
    """Mean outcome in the worst alpha tail (CVaR) — what a bad year actually looks like.

    Unlike VaR it is coherent (sub-additive), so it aggregates sensibly across a portfolio and
    can't be gamed by shifting mass just past the threshold.
    """
    x = _outcomes(outcomes)
    threshold = value_at_risk(x, alpha=alpha)
    tail = x[x <= threshold]
    return float(tail.mean())


def downside_deviation(outcomes: ArrayLike, *, target: float = 0.0) -> float:
    """RMS of shortfalls below ``target`` — volatility that only counts the bad side.

    Plain standard deviation punishes upside surprises too; this is the denominator of the
    Sortino ratio and the right spread measure for asymmetric outcome distributions.
    """
    x = _outcomes(outcomes)
    shortfall = np.minimum(x - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def max_drawdown(series: ArrayLike) -> float:
    """Worst peak-to-trough drop of a cumulative series, as a fraction of the peak.

    Path risk, not endpoint risk: two paths with equal final value can have very different
    survivability. Needs a strictly positive series (equity curve, cumulative revenue).
    """
    x = _outcomes(series)
    if np.any(x <= 0):
        raise ValueError("max_drawdown needs a strictly positive cumulative series")
    peaks = np.maximum.accumulate(x)
    return float(np.max(1.0 - x / peaks))


def probability_below(outcomes: ArrayLike, threshold: float) -> float:
    """Share of outcomes strictly below ``threshold`` — probability of failure / shortfall."""
    return float(np.mean(_outcomes(outcomes) < threshold))


def probability_above(outcomes: ArrayLike, threshold: float) -> float:
    """Share of outcomes at or above ``threshold`` — probability of hitting the target."""
    return float(np.mean(_outcomes(outcomes) >= threshold))


def sharpe_ratio(
    returns: ArrayLike, *, risk_free: float = 0.0, periods_per_year: float | None = None
) -> float:
    """Mean excess return per unit of volatility; ``periods_per_year`` annualizes (√t rule)."""
    x = _outcomes(returns) - risk_free
    spread = float(x.std(ddof=1))
    if spread == 0:
        return float("nan")
    ratio = float(x.mean()) / spread
    return ratio * float(np.sqrt(periods_per_year)) if periods_per_year else ratio


def sortino_ratio(
    returns: ArrayLike, *, target: float = 0.0, periods_per_year: float | None = None
) -> float:
    """Mean excess return per unit of *downside* deviation — Sharpe for asymmetric returns."""
    x = _outcomes(returns)
    downside = downside_deviation(x, target=target)
    if downside == 0:
        return float("nan")
    ratio = (float(x.mean()) - target) / downside
    return ratio * float(np.sqrt(periods_per_year)) if periods_per_year else ratio


def risk_summary(
    outcomes: ArrayLike, *, targets: ArrayLike | None = None, alpha: float = 0.05
) -> pl.DataFrame:
    """The one-table risk read: center, spread, P10/P50/P90, VaR/CVaR, target probabilities.

    The standard deck slide for a simulated business case — P50 is the plan, P10 the funding
    case, ``prob ≥ target`` the commitment you can defend.
    """
    x = _outcomes(outcomes)
    p5, p10, p50, p90, p95 = np.quantile(x, [0.05, 0.10, 0.50, 0.90, 0.95])
    rows = [
        {"metric": "mean", "value": float(x.mean())},
        {"metric": "std", "value": float(x.std(ddof=1)) if x.size > 1 else 0.0},
        {"metric": "p5", "value": float(p5)},
        {"metric": "p10", "value": float(p10)},
        {"metric": "p50", "value": float(p50)},
        {"metric": "p90", "value": float(p90)},
        {"metric": "p95", "value": float(p95)},
        {"metric": f"var_{int((1 - alpha) * 100)}", "value": value_at_risk(x, alpha=alpha)},
        {"metric": f"cvar_{int((1 - alpha) * 100)}", "value": expected_shortfall(x, alpha=alpha)},
    ]
    for target in np.asarray(targets, dtype=float) if targets is not None else []:
        rows.append({"metric": f"prob ≥ {target:g}", "value": probability_above(x, float(target))})
    return pl.DataFrame(rows)
