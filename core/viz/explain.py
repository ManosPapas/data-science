"""Model-explainability charts.

``feature_importance`` / ``permutation_importance`` take precomputed (names, values);
``partial_dependence`` takes a fitted sklearn estimator + X; the ``shap_*`` charts own their own
rendering (they return a ``Figure``) and need the ``shap`` extra (``uv sync --extra explain``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.inspection import PartialDependenceDisplay

from core.viz.base import chart


@chart(title="Feature importance")
def feature_importance(
    ax: Axes, names: Sequence[str], importances: ArrayLike, *, top: int = 20
) -> None:
    """Horizontal bar of the ``top`` most important features."""
    values = np.asarray(importances, dtype=float)
    order = np.argsort(values)[::-1][:top]
    sns.barplot(x=values[order], y=[names[int(i)] for i in order], ax=ax, color="tab:blue")
    ax.set(xlabel="importance", ylabel="")


@chart(title="Permutation importance")
def permutation_importance(
    ax: Axes,
    names: Sequence[str],
    importance_mean: ArrayLike,
    importance_std: ArrayLike,
    *,
    top: int = 20,
) -> None:
    """Permutation importance with error bars. Pass precomputed means/std (sklearn's
    ``permutation_importance`` result)."""
    mean = np.asarray(importance_mean, dtype=float)
    std = np.asarray(importance_std, dtype=float)
    order = np.argsort(mean)[::-1][:top]
    positions = np.arange(order.size)
    ax.barh(positions, mean[order], xerr=std[order], color="tab:blue")
    ax.set_yticks(positions, labels=[names[int(i)] for i in order])
    ax.invert_yaxis()
    ax.set(xlabel="importance (mean decrease)", ylabel="")


@chart(title="Partial dependence")
def partial_dependence(
    ax: Axes, estimator: Any, X: Any, features: Sequence[int | str], *, kind: str = "average"
) -> None:
    """PDP (``kind='average'``) or ICE (``kind='individual'``/``'both'``) for ``features``."""
    PartialDependenceDisplay.from_estimator(estimator, X, features, kind=kind, ax=ax)


def _shap_figure(render: Callable[[], object], save: str | None) -> Figure:
    """Run a SHAP plotting call (which draws on the current figure) and return that Figure."""
    import matplotlib.pyplot as plt

    render()
    figure = plt.gcf()
    if save:
        figure.savefig(save, bbox_inches="tight", dpi=150)
    return figure


def shap_summary(shap_values: Any, features: Any, *, save: str | None = None) -> Figure:
    """SHAP beeswarm summary. Needs the optional ``shap`` dependency; returns a Figure."""
    import shap

    return _shap_figure(lambda: shap.summary_plot(shap_values, features, show=False), save)


def shap_bar(shap_values: Any, features: Any, *, save: str | None = None) -> Figure:
    """Mean |SHAP| importance bar. Needs the optional ``shap`` dependency; returns a Figure."""
    import shap

    return _shap_figure(
        lambda: shap.summary_plot(shap_values, features, plot_type="bar", show=False), save
    )


def shap_dependence(
    feature: int | str, shap_values: Any, features: Any, *, save: str | None = None
) -> Figure:
    """SHAP dependence plot for one feature. Needs the ``shap`` extra; returns a Figure."""
    import shap

    return _shap_figure(
        lambda: shap.dependence_plot(feature, shap_values, features, show=False), save
    )


def shap_waterfall(explanation: Any, *, save: str | None = None) -> Figure:
    """SHAP waterfall for one prediction (pass a ``shap.Explanation``); returns a Figure."""
    import shap

    return _shap_figure(lambda: shap.plots.waterfall(explanation, show=False), save)
