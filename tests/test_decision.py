"""Tests for decision.bandits, decision.optimize, and decision.scenario."""

from __future__ import annotations

import numpy as np
import polars as pl

from core.decision import bandits, optimize, scenario


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
