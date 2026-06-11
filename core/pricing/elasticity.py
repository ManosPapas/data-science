"""Price elasticity of demand — constant-elasticity (log-log) demand estimation.

Fit ``ln(quantity) = intercept + elasticity * ln(price)``; the slope is the price elasticity
(typically negative — demand falls as price rises). Feed the fitted ``(intercept, elasticity)`` to
``core.pricing.optimize`` to find a revenue- or profit-maximizing price.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_demand(price: ArrayLike, quantity: ArrayLike) -> tuple[float, float]:
    """Fit constant-elasticity demand by OLS on logs; return ``(intercept, elasticity)``.

    ``price`` and ``quantity`` must be strictly positive (the fit is on their logs) — filter out
    zero-price/zero-sales rows first.
    """
    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("fit_demand requires strictly positive price and quantity")
    elasticity, intercept = np.polyfit(np.log(p), np.log(q), 1)
    return float(intercept), float(elasticity)


def price_elasticity(price: ArrayLike, quantity: ArrayLike) -> float:
    """Point price elasticity of demand (log-log slope); < -1 is elastic, -1..0 inelastic."""
    return fit_demand(price, quantity)[1]


def predict_demand(intercept: float, elasticity: float, price: ArrayLike) -> NDArray[np.float64]:
    """Predicted quantity at ``price`` under the fitted constant-elasticity model."""
    p = np.asarray(price, dtype=float)
    return np.asarray(np.exp(intercept) * p**elasticity, dtype=float)
