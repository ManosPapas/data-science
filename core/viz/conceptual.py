"""Illustrative charts — functions of a parameter, not your data (teaching/intuition aids)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from core.viz.base import chart


@chart(title="Gini impurity vs entropy")
def gini_vs_entropy(ax: Axes) -> None:
    """Impurity criteria for a binary split as a function of the positive-class probability."""
    p = np.linspace(0.001, 0.999, 200)
    gini = 2 * p * (1 - p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2  # scaled to [0, 0.5]
    misclassification = 1 - np.maximum(p, 1 - p)
    ax.plot(p, gini, label="Gini impurity")
    ax.plot(p, entropy, label="entropy (scaled)")
    ax.plot(p, misclassification, label="misclassification")
    ax.set(xlabel="p(class 1)", ylabel="impurity")
    ax.legend(loc="lower center")


@chart(title="Bias-variance tradeoff")
def bias_variance(ax: Axes) -> None:
    """Illustrative decomposition of total error into bias^2 and variance vs model complexity."""
    complexity = np.linspace(0, 10, 200)
    bias_squared = np.exp(-0.4 * complexity)
    variance = np.exp(0.3 * (complexity - 10))
    total = bias_squared + variance + 0.1
    ax.plot(complexity, bias_squared, label="bias^2")
    ax.plot(complexity, variance, label="variance")
    ax.plot(complexity, total, label="total error", linewidth=2)
    ax.axvline(complexity[int(np.argmin(total))], linestyle="--", color="grey")
    ax.set(xlabel="model complexity", ylabel="error")
    ax.legend(loc="upper center")
