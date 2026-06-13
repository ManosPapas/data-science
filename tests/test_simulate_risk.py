"""Tests for decision.simulate (Monte Carlo) and analytics.risk."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from core.analytics import risk
from core.decision import simulate


def test_monte_carlo_recovers_known_mean() -> None:
    result = simulate.monte_carlo(
        lambda volume, margin: volume * margin,
        {"volume": sps.norm(1000.0, 50.0), "margin": 2.0},
        n=20_000,
    )
    assert abs(result.mean - 2000.0) < 5.0
    assert result.p10 < result.p50 < result.p90
    assert result.prob_above(result.p10) == pytest.approx(0.9, abs=0.01)
    summary = result.summary(targets=[1900.0])
    assert summary.filter(summary["metric"] == "prob ≥ 1900")["value"][0] > 0.5


def test_monte_carlo_accepts_callable_and_scalar_loop() -> None:
    result = simulate.monte_carlo(
        lambda demand, price: demand * price,
        {"demand": lambda rng, n: rng.poisson(100.0, n).astype(float), "price": 5.0},
        n=500,
        vectorized=False,
    )
    assert abs(result.mean - 500.0) < 20.0


def test_monte_carlo_correlation_couples_inputs() -> None:
    result = simulate.monte_carlo(
        lambda a, b: a + b,
        {"a": sps.norm(0.0, 1.0), "b": sps.norm(0.0, 1.0)},
        n=5000,
        correlation={("a", "b"): 0.9},
    )
    observed = np.corrcoef(result.input_samples["a"], result.input_samples["b"])[0, 1]
    assert observed > 0.85
    # correlated inputs widen the spread of the sum vs independence (sqrt(2) -> ~1.95 sigma)
    assert result.std > 1.8


def test_monte_carlo_correlation_validations() -> None:
    with pytest.raises(ValueError, match="distribution inputs"):
        simulate.monte_carlo(
            lambda a, b: a + b,
            {"a": sps.norm(0, 1), "b": lambda rng, n: rng.normal(0, 1, n)},
            correlation={("a", "b"): 0.5},
        )
    with pytest.raises(ValueError, match="between -1 and 1"):
        simulate.monte_carlo(
            lambda a, b: a + b,
            {"a": sps.norm(0, 1), "b": sps.norm(0, 1)},
            correlation={("a", "b"): 1.0},
        )


def test_monte_carlo_drivers_ranks_dominant_input() -> None:
    result = simulate.monte_carlo(
        lambda big, small: big + 0.01 * small,
        {"big": sps.norm(0.0, 10.0), "small": sps.norm(0.0, 1.0)},
        n=4000,
    )
    drivers = result.drivers()
    assert drivers["input"][0] == "big"


def test_stress_test_includes_combined_row() -> None:
    def value(volume: float, price: float) -> float:
        return volume * price

    table = simulate.stress_test(
        value,
        {"volume": 100.0, "price": 10.0},
        {"demand shock": {"volume": 80.0}, "price war": {"price": 8.0}},
    )
    assert table["scenario"].to_list() == ["base", "demand shock", "price war", "combined"]
    combined = table.filter(table["scenario"] == "combined")
    assert combined["value"][0] == pytest.approx(640.0)
    assert combined["vs_base"][0] == pytest.approx(-360.0)


def test_simulate_paths_and_percentiles() -> None:
    paths = simulate.simulate_paths(start=100.0, drift=0.01, volatility=0.05, periods=12, n=500)
    assert paths.shape == (500, 13)
    assert np.all(paths[:, 0] == 100.0)
    bands = simulate.path_percentiles(paths)
    assert {"period", "p10", "p50", "p90"} == set(bands.columns)
    assert bands["p10"].to_list()[-1] < bands["p90"].to_list()[-1]

    additive = simulate.simulate_paths(
        start=0.0, drift=1.0, volatility=0.0, periods=3, n=2, model="additive"
    )
    assert additive[0].tolist() == [0.0, 1.0, 2.0, 3.0]


# --- analytics.risk ------------------------------------------------------------------------------


def test_var_and_expected_shortfall(rng: np.random.Generator) -> None:
    outcomes = rng.normal(0.0, 1.0, 50_000)
    var_5 = risk.value_at_risk(outcomes, alpha=0.05)
    assert abs(var_5 - (-1.645)) < 0.05
    cvar = risk.expected_shortfall(outcomes, alpha=0.05)
    assert cvar < var_5  # the tail mean is worse than its threshold
    assert abs(cvar - (-2.06)) < 0.08


def test_downside_deviation_ignores_upside() -> None:
    assert risk.downside_deviation([1.0, 2.0, 3.0], target=0.0) == 0.0
    assert risk.downside_deviation([-1.0, 1.0], target=0.0) == pytest.approx(np.sqrt(0.5))


def test_max_drawdown() -> None:
    assert risk.max_drawdown([100.0, 50.0, 120.0]) == pytest.approx(0.5)
    assert risk.max_drawdown([1.0, 2.0, 3.0]) == 0.0
    with pytest.raises(ValueError, match="positive"):
        risk.max_drawdown([1.0, -1.0])


def test_target_probabilities() -> None:
    outcomes = [0.0, 1.0, 2.0, 3.0]
    assert risk.probability_below(outcomes, 2.0) == 0.5
    assert risk.probability_above(outcomes, 2.0) == 0.5


def test_sharpe_and_sortino(rng: np.random.Generator) -> None:
    returns = rng.normal(0.01, 0.02, 2000)
    sharpe = risk.sharpe_ratio(returns)
    assert abs(sharpe - 0.5) < 0.1
    annual = risk.sharpe_ratio(returns, periods_per_year=12)
    assert annual == pytest.approx(sharpe * np.sqrt(12))
    assert risk.sortino_ratio(returns) > sharpe  # only downside vol in the denominator


def test_risk_summary_table(rng: np.random.Generator) -> None:
    outcomes = rng.normal(100.0, 10.0, 5000)
    table = risk.risk_summary(outcomes, targets=[90.0])
    metrics = dict(zip(table["metric"].to_list(), table["value"].to_list(), strict=True))
    assert abs(metrics["mean"] - 100.0) < 1.0
    assert metrics["p10"] < metrics["p50"] < metrics["p90"]
    assert metrics["prob ≥ 90"] > 0.8
    assert "var_95" in metrics
    assert "cvar_95" in metrics


# --- Regression tests for code-review fixes -----------------------------------------------------


def test_risk_summary_accepts_scalar_target_and_quantiles(rng: np.random.Generator) -> None:
    outcomes = rng.normal(100.0, 10.0, 2000)
    table = risk.risk_summary(outcomes, targets=90.0)  # scalar, was a TypeError
    assert "prob ≥ 90" in table["metric"].to_list()
    iqr = risk.risk_summary(outcomes, quantiles=(0.25, 0.75))
    assert {"p25", "p75"} <= set(iqr["metric"].to_list())
    assert "p50" not in iqr["metric"].to_list()


def test_simulation_summary_scalar_target_and_quantiles() -> None:
    from scipy import stats as sps

    result = simulate.monte_carlo(lambda a: a, {"a": sps.norm(0.0, 1.0)}, n=2000)
    table = result.summary(targets=0.0, quantiles=(5, 50, 95))
    metrics = table["metric"].to_list()
    assert {"p5", "p50", "p95"} <= set(metrics)
    assert "prob ≥ 0" in metrics
