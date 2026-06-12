"""Model-explainability charts.

``feature_importance`` / ``permutation_importance`` take precomputed (names, values);
``partial_dependence`` and the ``shap_*`` charts render multi-panel displays and return a
``Figure`` (shap needs the ``shap`` extra, ``uv sync --extra explain``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

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


def partial_dependence(
    estimator: Any,
    X: Any,
    features: Sequence[int | str],
    *,
    kind: str = "average",
    save: str | None = None,
) -> Figure:
    """PDP (``kind='average'``) or ICE (``'individual'``/``'both'``) for ``features``.

    sklearn's display draws child axes per feature (a single passed Axes would become an invisible
    bounding box), so this is multi-panel and returns a ``Figure``. Integer columns are coerced to
    float first — sklearn refuses integer features for PDP.
    """
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        ints = X.select_dtypes(include="integer").columns
        if len(ints):
            X = X.astype(dict.fromkeys(ints, "float64"))
    display = PartialDependenceDisplay.from_estimator(estimator, X, features, kind=kind)
    figure = cast(Figure, display.figure_)
    if save is not None:
        figure.savefig(save, bbox_inches="tight", dpi=150)
    return figure


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
