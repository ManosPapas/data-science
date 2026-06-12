"""Tests for analytics.curves (derivatives/turning points) and analytics.drivers (decomposition)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.analytics import curves, drivers


def test_slope_and_curvature_of_parabola() -> None:
    x = np.linspace(-5.0, 5.0, 201)
    y = x**2
    assert curves.slope(x, y)[100] == pytest.approx(0.0, abs=1e-9)  # vertex
    assert curves.slope(x, y)[150] == pytest.approx(2.0 * x[150], rel=0.01)
    assert np.allclose(curves.curvature(x, y)[5:-5], 2.0, atol=1e-6)


def test_point_elasticity_of_power_curve() -> None:
    x = np.linspace(1.0, 10.0, 200)
    y = 5.0 * x**3
    assert np.allclose(curves.point_elasticity(x, y), 3.0, atol=0.01)
    with pytest.raises(ValueError, match="positive"):
        curves.point_elasticity([-1.0, 1.0, 2.0], [1.0, 2.0, 3.0])


def test_local_extrema_finds_revenue_peak() -> None:
    price = np.linspace(1.0, 99.0, 99)
    revenue = price * (100.0 - price)  # peak at p = 50
    extrema = curves.local_extrema(price, revenue)
    assert extrema.height == 1
    assert extrema["kind"][0] == "maximum"
    assert extrema["x"][0] == pytest.approx(50.0, abs=0.5)


def test_inflection_points_of_s_curve() -> None:
    t = np.linspace(0.0, 10.0, 400)
    y = 1.0 / (1.0 + np.exp(-2.0 * (t - 5.0)))  # logistic, inflection at t=5
    inflections = curves.inflection_points(t, y)
    assert inflections.height >= 1
    closest = inflections.with_columns((pl.col("x") - 5.0).abs().alias("gap")).sort("gap")
    assert closest["x"][0] == pytest.approx(5.0, abs=0.2)
    assert closest["direction"][0] == "convex→concave"


def test_convexity_verdicts() -> None:
    x = np.linspace(-3.0, 3.0, 100)
    assert curves.convexity(x, x**2).verdict == "convex"
    assert curves.convexity(x, -(x**2)).verdict == "concave"
    assert curves.convexity(x, np.sin(3 * x)).verdict == "mixed"


def test_smooth_series_preserves_length_and_reduces_noise(rng: np.random.Generator) -> None:
    noisy = np.sin(np.linspace(0, 6, 200)) + rng.normal(0, 0.3, 200)
    smoothed = curves.smooth_series(noisy, window=9)
    assert smoothed.size == 200
    assert smoothed.std() < noisy.std()


def test_marginal_effect_and_gradient() -> None:
    def profit(price: float, volume: float) -> float:
        return price * volume

    base = {"price": 10.0, "volume": 100.0}
    assert curves.marginal_effect(profit, base, "price") == pytest.approx(100.0, rel=1e-4)
    grad = curves.gradient(profit, base)
    assert grad["volume"] == pytest.approx(10.0, rel=1e-4)


def test_response_curve_sweeps_one_input() -> None:
    def value(spend: float, base_sales: float) -> float:
        return base_sales + 50.0 * np.sqrt(spend)

    table = curves.response_curve(
        value, {"spend": 100.0, "base_sales": 0.0}, "spend", [25, 100, 400]
    )
    assert table["output"].to_list() == [250.0, 500.0, 1000.0]
    assert table["slope"][0] > table["slope"][2]  # diminishing returns


# --- drivers -------------------------------------------------------------------------------------


def test_change_decomposition_sums_to_total() -> None:
    baseline = pl.DataFrame({"region": ["EU", "NA", "UK"], "revenue": [100.0, 200.0, 50.0]})
    current = pl.DataFrame({"region": ["EU", "NA", "APAC"], "revenue": [90.0, 260.0, 30.0]})
    table = drivers.change_decomposition(current, baseline, value="revenue", by="region")
    assert table["change"].sum() == pytest.approx(380.0 - 350.0)
    assert table["share_of_change"].sum() == pytest.approx(1.0)
    apac = table.filter(pl.col("region") == "APAC")  # entrant counted in full
    assert apac["change"][0] == pytest.approx(30.0)


def test_price_volume_mix_bridge_is_exact(rng: np.random.Generator) -> None:
    segments = ["basic", "premium", "enterprise"]
    baseline = pl.DataFrame(
        {
            "segment": segments,
            "price": rng.uniform(50, 200, 3),
            "volume": rng.uniform(100, 1000, 3),
        }
    )
    current = pl.DataFrame(
        {
            "segment": segments,
            "price": rng.uniform(50, 200, 3),
            "volume": rng.uniform(100, 1000, 3),
        }
    )
    bridge = drivers.price_volume_mix(
        current, baseline, price="price", volume="volume", by="segment"
    )
    actual_change = float(bridge["revenue_1"].sum() - bridge["revenue_0"].sum())
    assert bridge["total_effect"].sum() == pytest.approx(actual_change)
    explained = float(
        (bridge["price_effect"] + bridge["volume_effect"] + bridge["mix_effect"]).sum()
    )
    assert explained == pytest.approx(actual_change)  # no residual


def test_price_volume_mix_pure_price_move() -> None:
    baseline = pl.DataFrame({"s": ["a", "b"], "price": [10.0, 20.0], "volume": [100.0, 100.0]})
    current = pl.DataFrame({"s": ["a", "b"], "price": [11.0, 22.0], "volume": [100.0, 100.0]})
    bridge = drivers.price_volume_mix(current, baseline, price="price", volume="volume", by="s")
    assert bridge["volume_effect"].sum() == pytest.approx(0.0, abs=1e-9)
    assert bridge["mix_effect"].sum() == pytest.approx(0.0, abs=1e-9)
    assert bridge["price_effect"].sum() == pytest.approx(300.0)


def test_revenue_leakage_ranks_leaks() -> None:
    df = pl.DataFrame(
        {
            "rep": ["ana", "ana", "ben", "ben"],
            "list_price": [100.0, 100.0, 100.0, 100.0],
            "realized": [100.0, 95.0, 70.0, 75.0],
        }
    )
    table = drivers.revenue_leakage(df, expected="list_price", actual="realized", by="rep")
    assert table["rep"][0] == "ben"
    assert table["leakage"][0] == pytest.approx(55.0)
    assert table["leakage_rate"][0] == pytest.approx(0.275)
    total = drivers.revenue_leakage(df, expected="list_price", actual="realized")
    assert total["leakage"][0] == pytest.approx(60.0)
