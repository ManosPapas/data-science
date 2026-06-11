"""Multi-armed & contextual bandits for online decisioning: pricing, allocation, ranking.

Each policy exposes ``select(...)`` to choose an arm and ``update(...)`` to learn from the reward —
pure numpy, no environment framework. Seeded RNGs keep runs reproducible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EpsilonGreedy:
    """Exploit the best arm; explore a random arm with probability ``epsilon``."""

    def __init__(self, n_arms: int, *, epsilon: float = 0.1, seed: int = 42) -> None:
        self.n_arms = n_arms
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self.counts: NDArray[np.float64] = np.zeros(n_arms)
        self.values: NDArray[np.float64] = np.zeros(n_arms)

    def select(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1.0
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonSampling:
    """Beta-Bernoulli Thompson sampling for binary (0/1) rewards."""

    def __init__(self, n_arms: int, *, seed: int = 42) -> None:
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)
        self.alpha: NDArray[np.float64] = np.ones(n_arms)
        self.beta: NDArray[np.float64] = np.ones(n_arms)

    def select(self) -> int:
        return int(np.argmax(self._rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


class UCB1:
    """Pick the arm with the highest upper-confidence bound on its mean reward."""

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.counts: NDArray[np.float64] = np.zeros(n_arms)
        self.values: NDArray[np.float64] = np.zeros(n_arms)
        self._t = 0

    def select(self) -> int:
        self._t += 1
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size:
            return int(unplayed[0])
        bonus = np.sqrt(2.0 * np.log(self._t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1.0
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class LinUCB:
    """Disjoint LinUCB — a contextual bandit with a linear reward model per arm over features."""

    def __init__(self, n_arms: int, n_features: int, *, alpha: float = 1.0) -> None:
        self.n_arms = n_arms
        self.alpha = alpha
        self.a: list[NDArray[np.float64]] = [np.eye(n_features) for _ in range(n_arms)]
        self.b: list[NDArray[np.float64]] = [np.zeros(n_features) for _ in range(n_arms)]

    def select(self, context: ArrayLike) -> int:
        x = np.asarray(context, dtype=float)
        scores: list[float] = []
        for arm in range(self.n_arms):
            a_inv = np.linalg.inv(self.a[arm])
            theta = a_inv @ self.b[arm]
            scores.append(float(theta @ x + self.alpha * np.sqrt(x @ a_inv @ x)))
        return int(np.argmax(scores))

    def update(self, arm: int, context: ArrayLike, reward: float) -> None:
        x = np.asarray(context, dtype=float)
        self.a[arm] += np.outer(x, x)
        self.b[arm] += reward * x
