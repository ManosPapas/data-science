"""Supply & demand analysis — equilibrium, censored demand, saturation, market structure.

Where ``pricing.demand`` models one side of the market, this module models the *meeting* of the
two: where price clears the market, how much demand a capacity cap hides (spill), how close a
market is to saturation, and how concentrated it is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Equilibrium:
    """Market-clearing price and the quantity traded there."""

    price: float
    quantity: float


def equilibrium(
    demand_fn: Callable[[float], float],
    supply_fn: Callable[[float], float],
    *,
    price_low: float,
    price_high: float,
) -> Equilibrium:
    """Find the price where demand equals supply (root of excess demand, Brent's method).

    Requires a shortage at ``price_low`` and a surplus at ``price_high`` (or vice versa) so a
    crossing exists in the bracket. Below the result the market is short; above it, oversupplied.
    """
    from scipy.optimize import brentq

    def excess(p: float) -> float:
        return float(demand_fn(p)) - float(supply_fn(p))

    lo, hi = excess(price_low), excess(price_high)
    if lo * hi > 0:
        raise ValueError("no equilibrium in bracket — demand and supply do not cross")
    p_star = float(brentq(excess, price_low, price_high))
    return Equilibrium(price=p_star, quantity=float(demand_fn(p_star)))


def linear_equilibrium(
    *,
    demand_intercept: float,
    demand_slope: float,
    supply_intercept: float,
    supply_slope: float,
) -> Equilibrium:
    """Closed-form equilibrium for linear demand and supply curves (q = a + b·p each)."""
    denominator = demand_slope - supply_slope
    if denominator == 0:
        raise ValueError("demand and supply slopes are equal — curves are parallel")
    price = (supply_intercept - demand_intercept) / denominator
    quantity = demand_intercept + demand_slope * price
    if price <= 0 or quantity <= 0:
        raise ValueError("curves cross at a non-positive price or quantity — check the signs")
    return Equilibrium(price=float(price), quantity=float(quantity))


def supply_demand_gap(
    demand: ArrayLike, supply: ArrayLike, *, labels: ArrayLike | None = None
) -> pl.DataFrame:
    """Per-period market balance: gap (excess demand), served volume, unmet demand, regime.

    Positive gap = shortage (lost sales / queues — scarcity you can price or expand into);
    negative = surplus (idle capacity / markdown pressure). Persistent one-sided gaps are the
    supply-demand mismatch signal.
    """
    d = np.asarray(demand, dtype=float)
    s = np.asarray(supply, dtype=float)
    if d.shape != s.shape:
        raise ValueError("demand and supply must have the same length")
    gap = d - s
    frame = pl.DataFrame(
        {
            "demand": d,
            "supply": s,
            "gap": gap,
            "served": np.minimum(d, s),
            "unmet": np.maximum(gap, 0.0),
            "regime": np.where(gap > 0, "shortage", np.where(gap < 0, "surplus", "balanced")),
        }
    )
    if labels is not None:
        frame = frame.with_columns(pl.Series("label", np.asarray(labels))).select(
            "label", "demand", "supply", "gap", "served", "unmet", "regime"
        )
    return frame


@dataclass(frozen=True)
class UnconstrainedDemand:
    """True-demand estimate recovered from capacity-capped sales (censored-normal MLE)."""

    mean: float
    std: float
    observed_mean: float
    spill: float
    spill_rate: float
    constrained_share: float


def unconstrain_demand(sales: ArrayLike, capacity: ArrayLike) -> UnconstrainedDemand:
    """Estimate true demand when you only observe ``sales = min(demand, capacity)``.

    Sold-out periods are right-censored: you know demand was *at least* capacity. Averaging raw
    sales therefore understates demand exactly where it matters. Treats demand as
    Normal(mean, std), fits by maximum likelihood, and prices the censoring: ``spill`` = average
    unmet demand per period, ``spill_rate`` = share of true demand you never served. The classic
    revenue-management "unconstraining" step before forecasting or capacity decisions.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    y = np.asarray(sales, dtype=float)
    c = np.asarray(capacity, dtype=float)
    if y.shape != c.shape:
        raise ValueError("sales and capacity must have the same length")
    if np.any(y > c + 1e-9):
        raise ValueError("sales exceed capacity — check the inputs")
    censored = y >= c - 1e-9
    if censored.all():
        raise ValueError("every period is at capacity — demand level is not identifiable")
    if not censored.any():
        mean, std = float(y.mean()), float(y.std(ddof=1))
        return UnconstrainedDemand(mean, std, mean, 0.0, 0.0, 0.0)

    observed = y[~censored]
    caps = c[censored]

    def negative_log_likelihood(params: NDArray[np.float64]) -> float:
        mu, log_sigma = params
        sigma = float(np.exp(log_sigma))
        uncensored = norm.logpdf(observed, loc=mu, scale=sigma).sum()
        right_tail = norm.logsf(caps, loc=mu, scale=sigma).sum()
        return -float(uncensored + right_tail)

    start = np.array([float(y.mean()), float(np.log(max(y.std(ddof=1), 1e-6)))])
    result = minimize(negative_log_likelihood, start, method="Nelder-Mead")
    if not result.success:
        raise ValueError(f"censored-demand MLE did not converge: {result.message}")
    mu, sigma = float(result.x[0]), float(np.exp(result.x[1]))

    # E[demand - cap | demand >= cap] for each sold-out period (truncated-normal mean above cap).
    alpha = (caps - mu) / sigma
    hazard = norm.pdf(alpha) / np.clip(norm.sf(alpha), 1e-12, None)
    unmet = sigma * (hazard - alpha)
    spill = float(unmet.sum() / y.size)
    total_true = float(y.sum() + unmet.sum())
    return UnconstrainedDemand(
        mean=mu,
        std=sigma,
        observed_mean=float(y.mean()),
        spill=spill,
        spill_rate=float(unmet.sum() / total_true) if total_true > 0 else 0.0,
        constrained_share=float(censored.mean()),
    )


@dataclass(frozen=True)
class SaturationFit:
    """Logistic (S-curve) growth fit — ``capacity`` is the implied market potential."""

    capacity: float
    midpoint: float
    rate: float
    r_squared: float

    def predict(self, t: ArrayLike) -> NDArray[np.float64]:
        """Fitted adoption/volume at time ``t``."""
        x = np.asarray(t, dtype=float)
        return np.asarray(
            self.capacity / (1.0 + np.exp(-self.rate * (x - self.midpoint))), dtype=float
        )

    def time_to_share(self, share: float) -> float:
        """When the market reaches ``share`` of potential (e.g. 0.9 → near-saturation date)."""
        if not 0.0 < share < 1.0:
            raise ValueError("share must be strictly between 0 and 1")
        return self.midpoint + float(np.log(share / (1.0 - share))) / self.rate


def saturation_fit(
    t: ArrayLike, y: ArrayLike, *, capacity_guess: float | None = None
) -> SaturationFit:
    """Fit logistic growth to cumulative adoption/volume — how much headroom is left?

    ``capacity`` minus the latest actual is the remaining market; growth investments stop paying
    once you price-in saturation. The estimate is weakly identified before the inflection point
    (curve still looks exponential) — treat early-stage capacity estimates as speculative.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(t, dtype=float)
    values = np.asarray(y, dtype=float)
    if x.size < 5:
        raise ValueError("need at least 5 points to fit an S-curve")

    def logistic(
        time: NDArray[np.float64], cap: float, mid: float, rate: float
    ) -> NDArray[np.float64]:
        return np.asarray(cap / (1.0 + np.exp(-rate * (time - mid))), dtype=float)

    span = float(x.max() - x.min()) or 1.0
    p0 = [capacity_guess or 1.2 * float(values.max()), float(np.median(x)), 4.0 / span]
    params, _ = curve_fit(logistic, x, values, p0=p0, maxfev=20_000)
    fitted = logistic(x, *params)
    tss = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((values - fitted) ** 2)) / tss if tss > 0 else float("nan")
    return SaturationFit(
        capacity=float(params[0]),
        midpoint=float(params[1]),
        rate=float(params[2]),
        r_squared=r_squared,
    )


def market_share(df: pl.DataFrame, *, value: str, by: str) -> pl.DataFrame:
    """Shares of ``value`` by player/segment, largest first, with the cumulative share."""
    totals = df.group_by(by).agg(pl.col(value).sum().alias("value")).sort("value", descending=True)
    return totals.with_columns(
        (pl.col("value") / pl.col("value").sum()).alias("share")
    ).with_columns(pl.col("share").cum_sum().alias("cumulative_share"))


def hhi(shares: ArrayLike) -> float:
    """Herfindahl-Hirschman concentration index on fractional shares (0-10,000 scale).

    Sum of squared percentage shares; the antitrust reading: < 1,500 competitive, 1,500-2,500
    moderately concentrated, > 2,500 concentrated. Shares are normalized to sum to 1 first.
    """
    s = np.asarray(shares, dtype=float)
    if np.any(s < 0) or s.sum() <= 0:
        raise ValueError("shares must be non-negative with a positive total")
    s = s / s.sum()
    return float(np.sum((100.0 * s) ** 2))
