"""Tests for decision.inventory (newsvendor/EOQ/safety stock) and decision.capacity (Erlang C)."""

from __future__ import annotations

import numpy as np
import pytest

from core.decision import capacity, inventory


def test_newsvendor_normal_overstocks_on_high_margin() -> None:
    result = inventory.newsvendor(
        price=10.0, cost=4.0, salvage=1.0, demand_mean=100.0, demand_std=20.0
    )
    assert result.critical_fractile == pytest.approx(6.0 / 9.0)
    assert result.quantity > 100.0  # high margin -> stock above mean demand
    assert result.expected_sales < 100.0
    assert result.expected_profit > 0


def test_newsvendor_samples_matches_quantile(rng: np.random.Generator) -> None:
    draws = rng.normal(100.0, 20.0, 20_000)
    result = inventory.newsvendor(price=10.0, cost=4.0, salvage=1.0, demand_samples=draws)
    assert result.quantity == pytest.approx(np.quantile(draws, 6.0 / 9.0))
    analytic = inventory.newsvendor(
        price=10.0, cost=4.0, salvage=1.0, demand_mean=100.0, demand_std=20.0
    )
    assert result.expected_profit == pytest.approx(analytic.expected_profit, rel=0.02)


def test_newsvendor_validations() -> None:
    with pytest.raises(ValueError, match="price must exceed cost"):
        inventory.newsvendor(price=4.0, cost=5.0, demand_mean=10.0, demand_std=1.0)
    with pytest.raises(ValueError, match="salvage"):
        inventory.newsvendor(price=10.0, cost=5.0, salvage=6.0, demand_mean=10.0, demand_std=1.0)


def test_eoq_closed_form() -> None:
    result = inventory.eoq(demand=1200.0, order_cost=100.0, holding_cost=2.0)
    assert result.order_quantity == pytest.approx(np.sqrt(120_000.0))
    assert result.total_cost == pytest.approx(np.sqrt(2 * 1200.0 * 100.0 * 2.0))
    assert result.orders_per_period == pytest.approx(1200.0 / result.order_quantity)


def test_safety_stock_and_reorder_point() -> None:
    buffer = inventory.safety_stock(
        demand_mean=50.0, demand_std=10.0, lead_time=4.0, service_level=0.95
    )
    assert buffer == pytest.approx(1.6449 * 10.0 * 2.0, rel=1e-3)  # z * sigma * sqrt(LT)
    point = inventory.reorder_point(
        demand_mean=50.0, demand_std=10.0, lead_time=4.0, service_level=0.95
    )
    assert point == pytest.approx(200.0 + buffer)
    # lead-time noise adds buffer
    noisy = inventory.safety_stock(
        demand_mean=50.0, demand_std=10.0, lead_time=4.0, lead_time_std=1.0
    )
    assert noisy > buffer


def test_simulate_inventory_policy_keeps_service() -> None:
    demand = np.full(50, 10.0)
    stock = inventory.simulate_inventory_policy(
        demand, reorder_at=30.0, order_quantity=50.0, lead_periods=2
    )
    assert stock.size == 50
    assert stock.min() >= 0  # deterministic demand + adequate reorder point -> no stockouts


def test_erlang_c_reduces_to_mm1() -> None:
    metrics = capacity.erlang_c(arrival_rate=1.0, service_rate=2.0, servers=1)
    assert metrics.utilization == pytest.approx(0.5)
    assert metrics.wait_probability == pytest.approx(0.5)  # M/M/1: P(wait) = rho
    assert metrics.average_wait == pytest.approx(0.5)  # C / (c*mu - lambda) = 0.5 / 1
    assert metrics.service_level(0.0) == pytest.approx(1.0 - metrics.wait_probability)
    assert metrics.service_level(10.0) > 0.99


def test_erlang_c_unstable_queue_raises() -> None:
    with pytest.raises(ValueError, match="unstable"):
        capacity.erlang_c(arrival_rate=10.0, service_rate=1.0, servers=5)


def test_required_servers_meets_target() -> None:
    sized = capacity.required_servers(
        arrival_rate=30.0, service_rate=4.0, target_wait_probability=0.2
    )
    assert sized.wait_probability <= 0.2
    one_less = capacity.erlang_c(arrival_rate=30.0, service_rate=4.0, servers=sized.servers - 1)
    assert one_less.wait_probability > 0.2 or one_less.utilization >= 1.0

    by_sla = capacity.required_servers(
        arrival_rate=30.0, service_rate=4.0, target_service_level=0.8, answer_within=0.05
    )
    assert by_sla.service_level(0.05) >= 0.8


def test_required_servers_needs_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        capacity.required_servers(arrival_rate=1.0, service_rate=2.0)


def test_queue_metrics_requires_rates() -> None:
    import pytest

    # the SLA-formula rates are now required — a hand-built QueueMetrics can't silently omit them
    with pytest.raises(TypeError):
        capacity.QueueMetrics(
            servers=3, utilization=0.5, wait_probability=0.2, average_wait=0.1, average_queue=0.3
        )
