"""Tests for decision.bandits, decision.optimize, and decision.scenario."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.decision import bandits, game, optimize, scenario


def test_epsilon_greedy_learns_best_arm() -> None:
    bandit = bandits.EpsilonGreedy(3, epsilon=0.1, seed=0)
    rewards = [0.1, 0.2, 0.9]  # arm 2 is best
    for _ in range(500):
        arm = bandit.select()
        bandit.update(arm, rewards[arm])
    assert int(np.argmax(bandit.values)) == 2


def test_thompson_learns_best_arm() -> None:
    ts = bandits.ThompsonSampling(3, seed=0)
    for _ in range(300):
        arm = ts.select()
        ts.update(arm, 1.0 if arm == 2 else 0.0)  # arm 2 always rewards
    picks = sum(ts.select() == 2 for _ in range(200))
    assert picks > 150  # converged onto the best arm


def test_ucb_selects_in_range() -> None:
    ucb = bandits.UCB1(4)
    arm = ucb.select()
    ucb.update(arm, 1.0)
    assert 0 <= ucb.select() < 4


def test_linucb_contextual() -> None:
    policy = bandits.LinUCB(3, n_features=2, alpha=1.0)
    context = [1.0, 0.5]
    arm = policy.select(context)
    assert 0 <= arm < 3
    policy.update(arm, context, 1.0)


def test_linear_program() -> None:
    # maximize x + y subject to x + y <= 4, x, y >= 0
    result = optimize.linear_program([1.0, 1.0], a_ub=[[1.0, 1.0]], b_ub=[4.0], maximize=True)
    assert result.success
    assert abs(float(result.x.sum()) - 4.0) < 1e-6


def test_assignment() -> None:
    _, cols = optimize.assign([[1.0, 2.0], [2.0, 1.0]])
    assert sorted(cols.tolist()) == [0, 1]


def test_expected_utility_and_certainty_equivalent() -> None:
    outcomes = [0.0, 100.0]
    probabilities = [0.5, 0.5]
    assert scenario.expected_utility(outcomes, probabilities) == 50.0  # a=0 -> plain EV
    ce = scenario.certainty_equivalent(outcomes, probabilities, risk_aversion=0.02)
    assert ce < 50.0  # risk-averse: the gamble is worth less than its mean
    assert scenario.certainty_equivalent(outcomes, probabilities) == 50.0


def test_scenario_table_values_what_ifs() -> None:
    def value(price: float, volume: float, cost: float) -> float:
        return (price - cost) * volume

    base = {"price": 10.0, "volume": 1000.0, "cost": 6.0}
    table = scenario.scenario_table(value, base, {"downturn": {"volume": 600.0}})
    assert table.filter(pl.col("scenario") == "base")["value"][0] == 4000.0
    assert table.filter(pl.col("scenario") == "downturn")["vs_base"][0] == -1600.0


def test_sensitivity_ranks_the_big_lever_first() -> None:
    def value(price: float, volume: float, cost: float) -> float:
        return (price - cost) * volume

    base = {"price": 10.0, "volume": 1000.0, "cost": 6.0}
    tornado = scenario.sensitivity(value, base, {"price": (9.0, 11.0), "cost": (5.8, 6.2)})
    assert tornado["input"][0] == "price"  # |2000| swing beats |-400|
    assert tornado.filter(pl.col("input") == "cost")["swing"][0] == -400.0


# --- Optimization extensions ---------------------------------------------------------------------


def test_integer_program_whole_units() -> None:
    # maximize 3x + 2y subject to x + y <= 1.5, integer -> x=1, y=0
    result = optimize.integer_program(
        [3.0, 2.0], a_ub=[[1.0, 1.0]], b_ub=[1.5], bounds=[(0, None), (0, None)], maximize=True
    )
    assert result.success
    assert result.x.tolist() == [1.0, 0.0]


def test_knapsack_exact() -> None:
    result = optimize.knapsack([60.0, 100.0, 120.0], [10.0, 20.0, 30.0], capacity=50.0)
    assert sorted(result.chosen) == [1, 2]
    assert result.total_value == 220.0
    assert result.total_weight == 50.0


def test_nonlinear_program_maximizes_concave() -> None:
    result = optimize.nonlinear_program(
        lambda x: -((x[0] - 3.0) ** 2), [0.0], bounds=[(0.0, 10.0)], maximize=True
    )
    assert result.success
    assert abs(float(result.x[0]) - 3.0) < 1e-4


def test_portfolio_weights_diversify() -> None:
    mu = np.array([0.10, 0.10])
    sigma = np.array([[0.04, 0.0], [0.0, 0.04]])
    result = optimize.portfolio_weights(mu, sigma, risk_aversion=4.0)
    assert np.allclose(result.weights, [0.5, 0.5], atol=1e-4)  # symmetric case splits evenly
    tilted = optimize.portfolio_weights(np.array([0.14, 0.10]), sigma, risk_aversion=4.0)
    assert tilted.weights[0] > 0.6  # higher-return asset gets more, not everything
    assert tilted.weights.sum() == pytest.approx(1.0)


def test_pareto_front_drops_dominated() -> None:
    points = np.array([[10.0, 1.0], [8.0, 3.0], [9.0, 0.5]])  # third dominated by first
    mask = optimize.pareto_front(points, maximize=True)
    assert mask.tolist() == [True, True, False]
    # mixed directions: maximize margin, minimize risk
    mixed = np.array([[10.0, 5.0], [9.0, 1.0], [8.0, 4.0]])
    mask2 = optimize.pareto_front(mixed, maximize=[True, False])
    assert mask2.tolist() == [True, True, False]


def test_scenario_optimize_mean_and_worst() -> None:
    scenarios = [{"s": 1.0}, {"s": 2.0}, {"s": 6.0}]

    def value(x: np.ndarray, s: float) -> float:
        return -((x[0] - s) ** 2)

    mean_fit = optimize.scenario_optimize(value, [0.0], scenarios, bounds=[(0.0, 10.0)])
    assert abs(float(mean_fit.x[0]) - 3.0) < 1e-3  # mean of scenarios
    worst_fit = optimize.scenario_optimize(
        value, [0.0], scenarios, bounds=[(0.0, 10.0)], criterion="worst"
    )
    assert abs(float(worst_fit.x[0]) - 3.5) < 1e-2  # midpoint of extremes


def test_shadow_prices_value_the_binding_constraint() -> None:
    lp = optimize.linear_program(
        [1.0, 1.4], a_ub=[[1.0, 1.0]], b_ub=[10.0], bounds=[(0, None), (0, None)], maximize=True
    )
    table = optimize.shadow_prices(lp, names_ub=["budget"], maximize=True)
    row = table.filter(pl.col("constraint") == "budget")
    assert row["shadow_price"][0] == pytest.approx(1.4)  # one more euro earns the best margin
    assert row["slack"][0] == pytest.approx(0.0)


# --- Game theory ----------------------------------------------------------------------------------


def test_pure_nash_prisoners_dilemma() -> None:
    # strategies: 0 = hold price, 1 = discount
    row_payoff = [[10.0, 2.0], [12.0, 4.0]]
    col_payoff = [[10.0, 12.0], [2.0, 4.0]]
    assert game.pure_nash(row_payoff, col_payoff) == [(1, 1)]  # both discount
    assert game.iterated_dominance(row_payoff, col_payoff) == ([1], [1])


def test_mixed_nash_matching_pennies() -> None:
    row_payoff = [[1.0, -1.0], [-1.0, 1.0]]
    col_payoff = [[-1.0, 1.0], [1.0, -1.0]]
    assert game.pure_nash(row_payoff, col_payoff) == []
    p, q = game.mixed_nash_2x2(row_payoff, col_payoff)
    assert np.allclose(p, [0.5, 0.5])
    assert np.allclose(q, [0.5, 0.5])


def test_best_response_dynamics_converges() -> None:
    # linear price reactions: p_i = 10 + 0.5 * p_j -> fixed point at 20/20
    result = game.best_response_dynamics(
        [lambda a: 10.0 + 0.5 * a[1], lambda a: 10.0 + 0.5 * a[0]], start=[5.0, 30.0]
    )
    assert result.converged
    assert np.allclose(result.point, [20.0, 20.0], atol=1e-5)
    assert result.history.shape[1] == 2
