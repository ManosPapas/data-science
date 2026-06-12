"""Price optimization — the price that maximizes revenue or profit under a demand model.

Pairs with ``core.pricing.elasticity``: estimate ``(intercept, elasticity)`` from data, then search
or solve for the best price. ``optimal_price`` grid-searches (robust for any demand);
``markup_price`` is the closed-form constant-elasticity optimum.
"""

from __future__ import annotations

import numpy as np
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
