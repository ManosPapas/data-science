"""Tests for the pricing subpackage."""

from __future__ import annotations

import numpy as np
import pytest

from core.pricing import elasticity, optimize


def test_fit_demand_recovers_elasticity(rng: np.random.Generator) -> None:
    price = rng.uniform(5.0, 50.0, 1000)
    quantity = np.exp(8.0) * price ** (-2.0) * rng.lognormal(0.0, 0.03, 1000)
    intercept, e = elasticity.fit_demand(price, quantity)
    assert abs(e - (-2.0)) < 0.1
    assert elasticity.predict_demand(intercept, e, np.array([10.0]))[0] > 0


def test_markup_price() -> None:
    assert optimize.markup_price(-2.0, 10.0) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="elastic"):
        optimize.markup_price(-0.5, 10.0)


def test_optimal_price_in_range() -> None:
    candidates = np.linspace(11.0, 40.0, 100)
    price, profit = optimize.optimal_price(8.0, -2.0, candidates, unit_cost=10.0)
    assert 11.0 <= price <= 40.0
    expected = float(optimize.profit_at(8.0, -2.0, np.array([price]), unit_cost=10.0)[0])
    assert profit == pytest.approx(expected)
