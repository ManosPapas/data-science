"""Price optimization — revenue/profit-maximizing prices, marginal economics, dynamic pricing.

Pairs with ``core.pricing.elasticity`` / ``core.pricing.demand``: estimate a demand model from
data, then search or solve for the best price. ``optimal_price`` grid-searches (robust for any
demand); ``markup_price`` / ``optimal_price_linear`` are the closed forms; ``dynamic_prices``
plans a price path for a fixed stock over a finite horizon (revenue management).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from core.pricing.elasticity import predict_demand


def revenue_at(intercept: float, elasticity: float, price: ArrayLike) -> NDArray[np.float64]:
    """Revenue ``price * demand(price)`` — the zero-cost case of :func:`profit_at`."""
    return profit_at(intercept, elasticity, price, unit_cost=0.0)


def profit_at(
    intercept: float, elasticity: float, price: ArrayLike, *, unit_cost: float
) -> NDArray[np.float64]:
    """Profit ``(price - unit_cost) * demand(price)`` under the fitted demand model."""
    p = np.asarray(price, dtype=float)
    return np.asarray((p - unit_cost) * predict_demand(intercept, elasticity, p), dtype=float)


def optimal_price(
    intercept: float, elasticity: float, candidates: ArrayLike, *, unit_cost: float = 0.0
) -> tuple[float, float]:
    """Grid-search ``candidates`` for the profit-maximizing price; returns ``(price, profit)``.

    With ``unit_cost=0`` this maximizes revenue. Robust for any demand shape — supply a realistic
    set of candidate price points to evaluate. Candidates must be finite and strictly positive
    (p=0 makes constant-elasticity demand blow up to inf, and argmax over NaN picks garbage).
    """
    prices = np.asarray(candidates, dtype=float)
    if prices.size == 0 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("candidates must be non-empty, finite, and strictly positive")
    profit = profit_at(intercept, elasticity, prices, unit_cost=unit_cost)
    if not np.all(np.isfinite(profit)):
        raise ValueError("demand model produced non-finite profit — check the fitted parameters")
    best = int(np.argmax(profit))
    return float(prices[best]), float(profit[best])


def markup_price(elasticity: float, unit_cost: float) -> float:
    """Closed-form profit-maximizing price for constant-elasticity demand: ``c * e/(e+1)``.

    Requires elastic demand (``elasticity < -1``); otherwise profit has no interior optimum.
    """
    if elasticity >= -1.0:
        raise ValueError("markup_price needs elastic demand (elasticity < -1)")
    return unit_cost * elasticity / (elasticity + 1.0)


def optimal_price_linear(intercept: float, slope: float, *, unit_cost: float = 0.0) -> float:
    """Closed-form profit-maximizing price for linear demand: midway between cost and choke.

    For ``q = intercept + slope·p`` (slope < 0), profit is a parabola with vertex
    ``(choke + unit_cost) / 2`` — the linear-demand counterpart of :func:`markup_price`.
    """
    if slope >= 0:
        raise ValueError("linear demand needs a negative slope")
    choke = -intercept / slope
    if choke <= unit_cost:
        raise ValueError("choke price is below unit cost — no profitable price exists")
    return (choke + unit_cost) / 2.0


def marginal_revenue(elasticity: float, price: ArrayLike) -> NDArray[np.float64]:
    """Marginal revenue per extra unit sold under constant elasticity: ``p · (1 + 1/e)``.

    MR < price always (selling one more unit means a lower price on *all* units); MR = 0 exactly
    at e = -1, the revenue peak — pricing into |e| < 1 territory destroys revenue at the margin.
    """
    p = np.asarray(price, dtype=float)
    return np.asarray(p * (1.0 + 1.0 / elasticity), dtype=float)


def marginal_profit(
    elasticity: float, price: ArrayLike, *, unit_cost: float
) -> NDArray[np.float64]:
    """Marginal profit per extra unit: MR - unit cost; zero exactly at :func:`markup_price`.

    Positive → price is above the optimum (cut to sell more); negative → below it. The sign is
    the direction-of-adjustment signal even when the demand fit is too rough to trust the level.
    """
    return np.asarray(marginal_revenue(elasticity, price) - unit_cost, dtype=float)


@dataclass(frozen=True)
class DynamicPricingPolicy:
    """A solved finite-horizon pricing policy: price per (period, remaining inventory)."""

    prices: NDArray[np.float64]  # (periods, capacity+1); [t, r] = price with r units left
    value: NDArray[np.float64]  # (periods+1, capacity+1) expected future revenue
    expected_revenue: float

    def price_for(self, period: int, remaining: int) -> float:
        """The planned price in ``period`` with ``remaining`` units unsold."""
        return float(self.prices[period, remaining])

    def policy_frame(self) -> pl.DataFrame:
        """Long-format (period, remaining, price) view for heatmaps and joins."""
        periods, levels = self.prices.shape
        return pl.DataFrame(
            {
                "period": np.repeat(np.arange(periods), levels),
                "remaining": np.tile(np.arange(levels), periods),
                "price": self.prices.ravel(),
            }
        )


def dynamic_prices(
    demand_rate: Callable[[float, int], float],
    *,
    capacity: int,
    periods: int,
    candidates: ArrayLike,
    salvage: float = 0.0,
) -> DynamicPricingPolicy:
    """Optimal price path for a fixed stock over a finite horizon (backward-induction DP).

    ``demand_rate(price, period)`` is the *expected* demand in that period at that price; sales
    are Poisson around it, capped by remaining stock. Unsold units are worth ``salvage`` at the
    end. The solved policy shows the two classic revenue-management forces: prices fall as the
    deadline nears (perishable inventory) and rise when stock runs scarce.
    """
    from scipy.stats import poisson

    prices = np.asarray(candidates, dtype=float)
    if prices.size == 0 or np.any(prices <= 0) or not np.all(np.isfinite(prices)):
        raise ValueError("candidates must be non-empty, finite, and strictly positive")
    if capacity < 1 or periods < 1:
        raise ValueError("capacity and periods must be at least 1")

    value = np.zeros((periods + 1, capacity + 1))
    value[periods] = salvage * np.arange(capacity + 1)
    policy = np.full((periods, capacity + 1), np.nan)

    units = np.arange(capacity + 1)
    for t in range(periods - 1, -1, -1):
        for r in range(1, capacity + 1):
            best_value, best_price = -np.inf, float("nan")
            for p in prices:
                rate = float(demand_rate(float(p), t))
                if rate < 0:
                    raise ValueError("demand_rate returned a negative expected demand")
                pmf = poisson.pmf(units[:r], rate)  # sell k = 0..r-1
                tail = float(poisson.sf(r - 1, rate))  # sell out: k = r
                revenue = float(np.sum(pmf * (p * units[:r] + value[t + 1, r - units[:r]])))
                revenue += tail * (p * r + value[t + 1, 0])
                if revenue > best_value:
                    best_value, best_price = revenue, float(p)
            value[t, r] = best_value
            policy[t, r] = best_price
        value[t, 0] = value[t + 1, 0]
    return DynamicPricingPolicy(
        prices=policy, value=value, expected_revenue=float(value[0, capacity])
    )
