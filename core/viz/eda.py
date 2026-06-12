"""Exploratory-data-analysis charts. Each is a ``@chart`` function: prepared data in, an Axes out.

DataFrame charts take a ``pl.DataFrame`` + column names; sample charts take a 1-D array-like.
``pairplot`` is multi-panel, so it returns a ``Figure`` instead of an ``Axes``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp, probplot

from core.analytics import stats
from core.viz.base import chart


@chart(title="Distribution")
def histogram(ax: Axes, values: ArrayLike, *, bins: int = 30, kde: bool = True) -> None:
    """Histogram (optionally with a KDE overlay) of a 1-D numeric sample."""
    sns.histplot(x=np.asarray(values, dtype=float), bins=bins, kde=kde, ax=ax)
    ax.set(ylabel="count")


@chart(title="ECDF")
def ecdf(ax: Axes, sample: ArrayLike) -> None:
    """Empirical cumulative distribution of a 1-D sample."""
    values = np.sort(np.asarray(sample, dtype=float))
    cumulative = np.arange(1, values.size + 1) / values.size
    ax.step(values, cumulative, where="post")
    ax.set(xlabel="value", ylabel="cumulative probability")


@chart()
def scatter(ax: Axes, df: pl.DataFrame, x: str, y: str, *, hue: str | None = None) -> None:
    """Scatter of ``y`` against ``x``, optionally coloured by the categorical ``hue``."""
    sns.scatterplot(data=df.to_pandas(), x=x, y=y, hue=hue, ax=ax, alpha=0.6, edgecolor="none")
    ax.set_title(f"{y} vs {x}")


@chart()
def boxplot_by(ax: Axes, df: pl.DataFrame, value: str, group: str) -> None:
    """Box plot of ``value`` split by the categorical ``group``."""
    pdf = df.select(group, value).to_pandas()
    sns.boxplot(data=pdf, x=group, y=value, ax=ax)
    ax.set_title(f"{value} by {group}")


@chart()
def count_bar(ax: Axes, df: pl.DataFrame, column: str) -> None:
    """Bar chart of category frequencies in ``column`` (most frequent first)."""
    counts = df[column].value_counts(sort=True).to_pandas()
    sns.barplot(data=counts, x=column, y="count", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{column} counts")


@chart(title="Cross-tab")
def crosstab_heatmap(ax: Axes, df: pl.DataFrame, row: str, col: str) -> None:
    """Heatmap of counts for two categorical columns."""
    pdf = df.select(row, col).to_pandas()
    sns.heatmap(pd.crosstab(pdf[row], pdf[col]), annot=True, fmt="d", cmap="Blues", ax=ax)


@chart(title="Missingness")
def missingness_bar(ax: Axes, df: pl.DataFrame) -> None:
    """Bar chart of the percentage of nulls per column (most-missing first)."""
    miss = stats.missingness(df).to_pandas()
    sns.barplot(data=miss, x="pct_null", y="column", ax=ax, color="tab:red")
    ax.set(xlabel="% null", ylabel="")


@chart(title="Correlation matrix")
def correlation_heatmap(ax: Axes, df: pl.DataFrame) -> None:
    """Heatmap of the Pearson correlation among numeric columns."""
    pdf = df.to_pandas()
    sns.heatmap(
        pdf.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        ax=ax,
    )


@chart(title="Q-Q plot (normal)")
def qq(ax: Axes, sample: ArrayLike) -> None:
    """Quantile-quantile plot of ``sample`` against a normal distribution.

    For a regression residual Q-Q, call ``qq(y_true - y_pred)`` — no separate chart needed.
    """
    probplot(np.asarray(sample, dtype=float), dist="norm", plot=ax)


@chart(title="KS plot")
def ks(ax: Axes, a: ArrayLike, b: ArrayLike, *, labels: tuple[str, str] = ("A", "B")) -> None:
    """Empirical CDFs of two samples, with the Kolmogorov-Smirnov statistic in the legend.

    For a classifier KS, pass each class's score array: ``ks(scores[y==0], scores[y==1])``.
    """
    xa = np.sort(np.asarray(a, dtype=float))
    xb = np.sort(np.asarray(b, dtype=float))
    ya = np.arange(1, xa.size + 1) / xa.size
    yb = np.arange(1, xb.size + 1) / xb.size
    ax.step(xa, ya, where="post", label=labels[0])
    ax.step(xb, yb, where="post", label=labels[1])
    statistic, p_value = ks_2samp(xa, xb)
    ax.set(xlabel="value", ylabel="cumulative probability")
    ax.legend(title=f"KS = {statistic:.3f}   p = {p_value:.3g}")


def pairplot(
    df: pl.DataFrame, columns: Sequence[str] | None = None, *, hue: str | None = None
) -> Figure:
    """Scatter-matrix of numeric features. Multi-panel, so it returns a ``Figure``."""
    pdf = df.to_pandas()
    chosen = list(columns) if columns is not None else None
    if chosen is not None and hue is not None and hue not in chosen:
        chosen = [*chosen, hue]
    grid = sns.pairplot(pdf if chosen is None else pdf[chosen], hue=hue)
    return cast(Figure, grid.figure)


@chart(title="Fitted distribution")
def fit_overlay(
    ax: Axes,
    sample: ArrayLike,
    *,
    dist: str,
    params: Sequence[float] | Mapping[str, float],
) -> None:
    """Histogram with a fitted distribution overlaid — the eyeball check on an MLE fit.

    Continuous fits pass the ``params`` tuple from ``stats.fit_distribution`` (scipy order);
    discrete fits pass the ``params`` mapping from ``stats.fit_discrete`` (including ``'zip'``,
    which has no scipy twin). A good AIC with a visibly wrong tail is exactly what this chart
    exists to catch.
    """
    import scipy.stats as sps

    values = np.asarray(sample, dtype=float)
    if dist == "zip":  # zero-inflated Poisson — drawn from its mixture pmf directly
        if not isinstance(params, Mapping):
            raise ValueError("'zip' params come as the mapping from stats.fit_discrete")
        pi, mu = float(params["pi"]), float(params["mu"])
        observed, counts = np.unique(values.astype(int), return_counts=True)
        grid = np.arange(observed.min(), observed.max() + 1)
        pmf = np.where(
            grid == 0, pi + (1.0 - pi) * np.exp(-mu), (1.0 - pi) * sps.poisson.pmf(grid, mu)
        )
        ax.bar(observed, counts / values.size, width=0.8, alpha=0.45, label="observed")
        ax.plot(grid, pmf, "o-", color="#1a1a1a", label="zip fit")
        ax.set(xlabel="count", ylabel="probability")
        ax.legend()
        return
    model = getattr(sps, dist)
    frozen = model(**params) if isinstance(params, Mapping) else model(*params)
    if hasattr(frozen.dist, "pmf"):  # discrete: observed shares vs fitted pmf
        observed, counts = np.unique(values.astype(int), return_counts=True)
        grid = np.arange(observed.min(), observed.max() + 1)
        ax.bar(observed, counts / values.size, width=0.8, alpha=0.45, label="observed")
        ax.plot(grid, frozen.pmf(grid), "o-", color="#1a1a1a", label=f"{dist} fit")
        ax.set(xlabel="count", ylabel="probability")
    else:
        sns.histplot(x=values, stat="density", bins=40, alpha=0.45, ax=ax)
        grid = np.linspace(values.min(), values.max(), 300)
        ax.plot(grid, frozen.pdf(grid), color="#1a1a1a", label=f"{dist} fit")
        ax.set(xlabel="value", ylabel="density")
    ax.legend()
