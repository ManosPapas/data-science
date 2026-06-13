"""Tests for the pricing subpackage."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.pricing import demand, elasticity, market, optimize


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


def test_fit_demand_needs_price_variation(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="distinct prices"):
        elasticity.fit_demand(np.full(50, 9.99), rng.uniform(10.0, 20.0, 50))


def test_optimal_price_rejects_bad_candidates() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        optimize.optimal_price(8.0, -2.0, np.linspace(0.0, 50.0, 11))  # p=0 -> infinite demand
    with pytest.raises(ValueError, match="strictly positive"):
        optimize.optimal_price(8.0, -2.0, np.array([]))


def test_optimal_price_in_range() -> None:
    candidates = np.linspace(11.0, 40.0, 100)
    price, profit = optimize.optimal_price(8.0, -2.0, candidates, unit_cost=10.0)
    assert 11.0 <= price <= 40.0
    expected = float(optimize.profit_at(8.0, -2.0, np.array([price]), unit_cost=10.0)[0])
    assert profit == pytest.approx(expected)


# --- Elasticity extensions ------------------------------------------------------------------


def _loglog_data(
    rng: np.random.Generator, elasticity_true: float = -2.0, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    price = rng.uniform(5.0, 50.0, n)
    quantity = np.exp(8.0) * price**elasticity_true * rng.lognormal(0.0, 0.05, n)
    return price, quantity


def test_fit_demand_ci_brackets_truth(rng: np.random.Generator) -> None:
    price, quantity = _loglog_data(rng)
    fit = elasticity.fit_demand_ci(price, quantity)
    assert fit.ci_low < -2.0 < fit.ci_high
    assert fit.std_err > 0
    assert 0.9 < fit.r_squared <= 1.0
    assert fit.n == 400


def test_bootstrap_elasticity_brackets_truth(rng: np.random.Generator) -> None:
    price, quantity = _loglog_data(rng)
    low, high = elasticity.bootstrap_elasticity(price, quantity, n_boot=300, seed=1)
    assert low < -2.0 < high


def test_cross_price_elasticity_recovers_both(rng: np.random.Generator) -> None:
    own = rng.uniform(10.0, 40.0, 1500)
    rival = rng.uniform(10.0, 40.0, 1500)
    quantity = np.exp(6.0) * own**-2.0 * rival**0.8 * rng.lognormal(0.0, 0.05, 1500)
    table = elasticity.cross_price_elasticity(quantity, own, {"rival": rival})
    own_row = table.filter(pl.col("term") == "own")
    rival_row = table.filter(pl.col("term") == "rival")
    assert abs(own_row["elasticity"][0] - (-2.0)) < 0.1
    assert abs(rival_row["elasticity"][0] - 0.8) < 0.1
    assert rival_row["relationship"][0] == "substitute"


def test_segment_elasticity_orders_segments(rng: np.random.Generator) -> None:
    frames = []
    for name, e in [("loyal", -1.2), ("deal_hunters", -3.0)]:
        price, quantity = _loglog_data(rng, elasticity_true=e, n=300)
        frames.append(pl.DataFrame({"price": price, "quantity": quantity, "segment": name}))
    table = elasticity.segment_elasticity(
        pl.concat(frames), price="price", quantity="quantity", segment="segment"
    )
    assert table["segment"][0] == "deal_hunters"  # most negative first
    assert table.height == 2


def test_rolling_elasticity_shape(rng: np.random.Generator) -> None:
    price, quantity = _loglog_data(rng, n=200)
    df = pl.DataFrame({"p": price, "q": quantity, "t": np.arange(200)})
    rolled = elasticity.rolling_elasticity(df, price="p", quantity="q", time="t", window=60)
    assert rolled.height == 141  # 200 - 60 + 1
    assert {"elasticity", "ci_low", "ci_high", "n"} <= set(rolled.columns)


def test_elasticity_drift_detects_shift(rng: np.random.Generator) -> None:
    p1, q1 = _loglog_data(rng, elasticity_true=-1.5, n=300)
    p2, q2 = _loglog_data(rng, elasticity_true=-3.0, n=300)
    df = pl.DataFrame(
        {"p": np.concatenate([p1, p2]), "q": np.concatenate([q1, q2]), "t": np.arange(600)}
    )
    drift = elasticity.elasticity_drift(df, price="p", quantity="q", time="t")
    assert drift.drifted
    assert drift.recent.elasticity < drift.baseline.elasticity

    p3, q3 = _loglog_data(rng, elasticity_true=-1.5, n=300)
    stable = pl.DataFrame(
        {"p": np.concatenate([p1, p3]), "q": np.concatenate([q1, q3]), "t": np.arange(600)}
    )
    assert not elasticity.elasticity_drift(stable, price="p", quantity="q", time="t").drifted


def test_nonlinear_elasticity_check(rng: np.random.Generator) -> None:
    price, quantity = _loglog_data(rng)
    assert not elasticity.nonlinear_elasticity_check(price, quantity).nonlinear

    log_p = np.log(price)
    curved_q = np.exp(8.0 - 1.0 * log_p - 0.4 * log_p**2) * rng.lognormal(0.0, 0.05, 400)
    check = elasticity.nonlinear_elasticity_check(price, curved_q)
    assert check.nonlinear
    local = check.local_elasticity(np.array([10.0, 40.0]))
    assert local[1] < local[0]  # more elastic at higher prices


def test_elasticity_decomposition_sums_exactly() -> None:
    before = pl.DataFrame(
        {"segment": ["a", "b"], "elasticity": [-1.0, -2.0], "weight": [50.0, 50.0]}
    )
    after = pl.DataFrame(
        {"segment": ["a", "b"], "elasticity": [-1.5, -2.0], "weight": [30.0, 70.0]}
    )
    table = elasticity.elasticity_decomposition(before, after)
    aggregate_before = elasticity.aggregate_elasticity([-1.0, -2.0], [50.0, 50.0])
    aggregate_after = elasticity.aggregate_elasticity([-1.5, -2.0], [30.0, 70.0])
    assert table["total"].sum() == pytest.approx(aggregate_after - aggregate_before)


# --- Demand curves, WTP, purchase probability ---------------------------------------------------


def test_fit_linear_demand_recovers_curve(rng: np.random.Generator) -> None:
    price = rng.uniform(10.0, 90.0, 500)
    quantity = 1000.0 - 10.0 * price + rng.normal(0.0, 20.0, 500)
    fit = demand.fit_linear_demand(price, quantity)
    assert abs(fit.slope - (-10.0)) < 0.5
    assert abs(fit.choke_price - 100.0) < 2.0
    assert fit.predict(np.array([200.0]))[0] == 0.0  # floored past choke
    assert fit.elasticity_at(np.array([50.0]))[0] < -0.5


def test_logit_demand_and_wtp(rng: np.random.Generator) -> None:
    price = rng.uniform(50.0, 150.0, 4000)
    wtp = rng.logistic(100.0, 10.0, 4000)
    bought = (wtp >= price).astype(float)
    model = demand.fit_logit_demand(price, bought)
    assert abs(model.wtp_median - 100.0) < 3.0
    table = demand.willingness_to_pay(price, bought)
    values = table["wtp"].to_numpy()
    assert np.all(np.diff(values) > 0)  # quantiles increase
    probabilities = model.predict(np.array([60.0, 100.0, 140.0]))
    assert probabilities[0] > 0.9
    assert abs(probabilities[1] - 0.5) < 0.05


def test_van_westendorp_points_ordered(rng: np.random.Generator) -> None:
    center = rng.normal(100.0, 15.0, 500)
    result = demand.van_westendorp(
        too_cheap=center - 30.0,
        cheap=center - 15.0,
        expensive=center + 15.0,
        too_expensive=center + 30.0,
    )
    low, high = result.acceptable_range
    assert low < result.optimal_price < high
    assert abs(result.optimal_price - 100.0) < 5.0
    assert {"price", "too_cheap", "too_expensive"} <= set(result.curves.columns)


def test_demand_schedule() -> None:
    table = demand.demand_schedule(lambda p: 100.0 - p, np.array([10.0, 20.0]))
    assert table["revenue"].to_list() == [900.0, 1600.0]


# --- Market: equilibrium, censored demand, saturation, structure --------------------------------


def test_equilibrium_matches_closed_form() -> None:
    closed = market.linear_equilibrium(
        demand_intercept=100.0, demand_slope=-2.0, supply_intercept=10.0, supply_slope=1.0
    )
    assert closed.price == pytest.approx(30.0)
    assert closed.quantity == pytest.approx(40.0)
    numeric = market.equilibrium(
        lambda p: 100.0 - 2.0 * p, lambda p: 10.0 + p, price_low=1.0, price_high=50.0
    )
    assert numeric.price == pytest.approx(closed.price, abs=1e-6)


def test_supply_demand_gap_regimes() -> None:
    table = market.supply_demand_gap([120.0, 80.0, 100.0], [100.0, 100.0, 100.0])
    assert table["regime"].to_list() == ["shortage", "surplus", "balanced"]
    assert table["unmet"].to_list() == [20.0, 0.0, 0.0]


def test_unconstrain_demand_recovers_truth(rng: np.random.Generator) -> None:
    true_demand = rng.normal(100.0, 20.0, 800)
    capacity = np.full(800, 110.0)
    sales = np.minimum(true_demand, capacity)
    result = market.unconstrain_demand(sales, capacity)
    assert abs(result.mean - 100.0) < 3.0
    assert result.mean > result.observed_mean  # censoring bias corrected
    assert result.spill > 0
    assert 0 < result.constrained_share < 1


def test_saturation_fit_recovers_capacity(rng: np.random.Generator) -> None:
    t = np.arange(60, dtype=float)
    y = 5000.0 / (1.0 + np.exp(-0.25 * (t - 30.0))) + rng.normal(0.0, 30.0, 60)
    fit = market.saturation_fit(t, y)
    assert abs(fit.capacity - 5000.0) / 5000.0 < 0.05
    assert abs(fit.midpoint - 30.0) < 2.0
    assert fit.time_to_share(0.5) == pytest.approx(fit.midpoint)


def test_market_share_and_hhi() -> None:
    df = pl.DataFrame({"player": ["a", "a", "b", "c"], "revenue": [40.0, 20.0, 30.0, 10.0]})
    shares = market.market_share(df, value="revenue", by="player")
    assert shares["player"][0] == "a"
    assert shares["share"][0] == pytest.approx(0.6)
    assert market.hhi([1.0]) == pytest.approx(10_000.0)
    assert market.hhi([0.25] * 4) == pytest.approx(2_500.0)


# --- Pricing optimization extensions -------------------------------------------------------------


def test_marginal_revenue_and_profit() -> None:
    assert optimize.marginal_revenue(-1.0, np.array([50.0]))[0] == pytest.approx(0.0)
    best = optimize.markup_price(-2.0, 10.0)
    marginal = optimize.marginal_profit(-2.0, np.array([best]), unit_cost=10.0)
    assert marginal[0] == pytest.approx(0.0)


def test_optimal_price_linear_matches_grid() -> None:
    closed = optimize.optimal_price_linear(1000.0, -10.0, unit_cost=20.0)
    assert closed == pytest.approx(60.0)  # (choke 100 + cost 20) / 2
    grid = np.linspace(20.0, 100.0, 8001)
    profit = (grid - 20.0) * (1000.0 - 10.0 * grid)
    assert grid[int(np.argmax(profit))] == pytest.approx(closed, abs=0.02)


def test_dynamic_prices_scarcity_logic() -> None:
    def demand_rate(price: float, period: int) -> float:
        return 8.0 * (price / 100.0) ** -2.0  # elastic, time-invariant

    policy = optimize.dynamic_prices(
        demand_rate, capacity=20, periods=5, candidates=np.array([80.0, 100.0, 120.0, 140.0])
    )
    assert policy.expected_revenue > 0
    assert policy.prices.shape == (5, 21)
    # scarcer stock should never be priced lower at the same point in time
    assert np.all(np.diff(policy.prices[0, 1:]) <= 1e-9)
    assert policy.policy_frame().height == 5 * 21


def test_segment_elasticity_skips_null_segment(rng: np.random.Generator) -> None:
    # a segment carrying a null quantity must be skipped, not crash the whole loop
    good = _loglog_data(rng, elasticity_true=-2.0, n=200)
    frames = [
        pl.DataFrame({"price": good[0], "quantity": good[1], "segment": "clean"}),
        pl.DataFrame(
            {"price": [10.0, 20.0, 30.0], "quantity": [5.0, None, 3.0], "segment": "dirty"}
        ),
    ]
    table = elasticity.segment_elasticity(
        pl.concat(frames), price="price", quantity="quantity", segment="segment"
    )
    assert "clean" in table["segment"].to_list()
    assert "dirty" not in table["segment"].to_list()
