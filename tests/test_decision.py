"""Tests for decision.bandits and decision.optimize."""

from __future__ import annotations

import numpy as np

from core.decision import bandits, optimize


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
