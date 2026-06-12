"""Inventory optimization — newsvendor stocking, EOQ ordering, safety stock & reorder points.

Three classic, closed-form answers to "how much should we hold?": how much to stock when demand
is uncertain and the season is one shot (newsvendor), how much to order at a time when demand is
steady (EOQ), and how much buffer protects service through lead time (safety stock).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class NewsvendorResult:
    """Optimal one-shot stocking decision under demand uncertainty."""

    quantity: float
    critical_fractile: float
    expected_profit: float
    expected_sales: float
    expected_leftover: float


def newsvendor(
    *,
    price: float,
    cost: float,
    salvage: float = 0.0,
    demand_mean: float | None = None,
    demand_std: float | None = None,
    demand_samples: ArrayLike | None = None,
) -> NewsvendorResult:
    """Profit-maximizing stock for one selling season: balance stock-outs against leftovers.

    The optimum covers demand with probability = critical fractile ``(price-cost)/(price-salvage)``
    — *not* mean demand: high margins justify deliberate overstock, thin margins deliberate
    stock-outs. Give either a Normal demand (``demand_mean``/``demand_std``) or empirical
    ``demand_samples`` (e.g. a forecast's Monte Carlo draws).
    """
    if not cost < price:
        raise ValueError("price must exceed cost — otherwise selling is pointless")
    if not salvage < cost:
        raise ValueError("salvage must be below cost — otherwise overstocking is free")
    fractile = (price - cost) / (price - salvage)

    if demand_samples is not None:
        draws = np.asarray(demand_samples, dtype=float)
        if draws.size < 2:
            raise ValueError("need at least 2 demand samples")
        quantity = float(np.quantile(draws, fractile))
        sales = np.minimum(draws, quantity)
        leftover = quantity - sales
        profit = price * sales + salvage * leftover - cost * quantity
        return NewsvendorResult(
            quantity=quantity,
            critical_fractile=float(fractile),
            expected_profit=float(profit.mean()),
            expected_sales=float(sales.mean()),
            expected_leftover=float(leftover.mean()),
        )

    if demand_mean is None or demand_std is None:
        raise ValueError("provide demand_mean and demand_std, or demand_samples")
    if demand_std <= 0:
        raise ValueError("demand_std must be positive")
    from scipy.stats import norm

    quantity = float(norm.ppf(fractile, loc=demand_mean, scale=demand_std))
    z = (quantity - demand_mean) / demand_std
    # Expected unmet demand E[(D-q)+] = sigma * (pdf(z) - z * sf(z)) — the normal loss function.
    expected_short = demand_std * float(norm.pdf(z) - z * norm.sf(z))
    expected_sales = demand_mean - expected_short
    expected_leftover = quantity - expected_sales
    expected_profit = price * expected_sales + salvage * expected_leftover - cost * quantity
    return NewsvendorResult(
        quantity=quantity,
        critical_fractile=float(fractile),
        expected_profit=float(expected_profit),
        expected_sales=float(expected_sales),
        expected_leftover=float(expected_leftover),
    )


@dataclass(frozen=True)
class EoqResult:
    """Economic order quantity with its ordering cadence and total holding+ordering cost."""

    order_quantity: float
    orders_per_period: float
    total_cost: float


def eoq(*, demand: float, order_cost: float, holding_cost: float) -> EoqResult:
    """Economic order quantity ``√(2·D·S/H)`` — the batch size balancing ordering vs holding.

    ``demand`` (units/period), ``order_cost`` (per order placed), ``holding_cost`` (per unit per
    period — typically 15-30% of unit value/year). The optimum is famously flat: being 20% off
    on Q costs ~2% — get H roughly right and move on.
    """
    if min(demand, order_cost, holding_cost) <= 0:
        raise ValueError("demand, order_cost, and holding_cost must all be positive")
    quantity = float(np.sqrt(2.0 * demand * order_cost / holding_cost))
    return EoqResult(
        order_quantity=quantity,
        orders_per_period=demand / quantity,
        total_cost=float(np.sqrt(2.0 * demand * order_cost * holding_cost)),
    )


def safety_stock(
    *,
    demand_mean: float,
    demand_std: float,
    lead_time: float,
    service_level: float = 0.95,
    lead_time_std: float = 0.0,
) -> float:
    """Buffer stock for the target cycle service level over (possibly noisy) lead time.

    ``z · √(LT·sd_demand² + mean_demand²·sd_leadtime²)`` — demand noise and lead-time noise both eat
    service. The cost of service is convex in the level: 95% → 99% costs far more than 90% → 95%,
    so pick service levels per item value, not one global number.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be strictly between 0 and 1")
    if lead_time <= 0 or demand_std < 0 or lead_time_std < 0:
        raise ValueError("lead_time must be positive; deviations non-negative")
    from scipy.stats import norm

    z = float(norm.ppf(service_level))
    variance = lead_time * demand_std**2 + (demand_mean**2) * lead_time_std**2
    return z * float(np.sqrt(variance))


def reorder_point(
    *,
    demand_mean: float,
    demand_std: float,
    lead_time: float,
    service_level: float = 0.95,
    lead_time_std: float = 0.0,
) -> float:
    """Order when stock falls to expected lead-time demand plus :func:`safety_stock`."""
    buffer = safety_stock(
        demand_mean=demand_mean,
        demand_std=demand_std,
        lead_time=lead_time,
        service_level=service_level,
        lead_time_std=lead_time_std,
    )
    return demand_mean * lead_time + buffer


def simulate_inventory_policy(
    demand: ArrayLike, *, reorder_at: float, order_quantity: float, lead_periods: int = 1
) -> NDArray[np.float64]:
    """Stock level per period under an (R, Q) policy against a demand series — policy check.

    Replays history (or simulated paths): order ``order_quantity`` whenever stock ends a period
    at/below ``reorder_at``; arrivals land after ``lead_periods``. Negative stock = stock-out
    depth. Validates the closed forms above against lumpy real demand before you trust them.
    """
    d = np.asarray(demand, dtype=float)
    if lead_periods < 0:
        raise ValueError("lead_periods must be non-negative")
    stock = np.empty(d.size)
    level = reorder_at + order_quantity  # start fully replenished
    arrivals = np.zeros(d.size + lead_periods + 1)
    on_order = 0.0
    for t in range(d.size):
        level += arrivals[t]
        on_order -= arrivals[t]
        level -= d[t]
        if level + on_order <= reorder_at:
            arrivals[t + lead_periods + 1] += order_quantity
            on_order += order_quantity
        stock[t] = level
    return stock
