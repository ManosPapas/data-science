"""Illustrative charts — functions of a parameter, not your data (teaching/intuition aids)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


@chart(title="Causal graph")
def dag(
    ax: Axes,
    edges: Sequence[tuple[str, str]],
    *,
    positions: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    """Draw a causal DAG from (cause, effect) edges — the adjust-for-what reasoning aid.

    Nodes are laid out left-to-right by causal depth (or pass explicit ``positions``). Use it to
    pick the adjustment set before a causal estimate: block backdoor paths (confounders), and do
    *not* condition on colliders or on mediators of the effect you want. Drawing only — it does
    not verify acyclicity.
    """
    nodes = list(dict.fromkeys(node for edge in edges for node in edge))
    if positions is None:
        depth = dict.fromkeys(nodes, 0)
        for _ in nodes:  # relax repeatedly: depth = longest cause-chain (cycle-safe bound)
            for cause, effect in edges:
                depth[effect] = max(depth[effect], depth[cause] + 1)
        levels: dict[int, list[str]] = {}
        for node in nodes:
            levels.setdefault(depth[node], []).append(node)
        positions = {
            node: (float(level), (len(members) - 1) / 2 - float(i))
            for level, members in levels.items()
            for i, node in enumerate(members)
        }
    for cause, effect in edges:
        ax.annotate(
            "",
            xy=positions[effect],
            xytext=positions[cause],
            arrowprops={"arrowstyle": "-|>", "color": "grey", "shrinkA": 20, "shrinkB": 20},
        )
    for node, (x, y) in positions.items():
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "tab:blue"},
        )
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_axis_off()


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
