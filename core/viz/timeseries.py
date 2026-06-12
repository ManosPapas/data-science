"""Time-series & commercial charts.

Forecast/cohort/rolling charts draw on a single Axes; ``seasonal_decomposition`` is multi-panel and
returns a ``Figure``. statsmodels is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from core.viz.base import chart


@chart(title="Forecast vs actual")
def forecast(
    ax: Axes,
    dates: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
) -> None:
    """Actual vs forecast over time, with an optional prediction-interval band."""
    x = np.asarray(dates)
    ax.plot(x, np.asarray(actual, dtype=float), label="actual")
    ax.plot(x, np.asarray(predicted, dtype=float), label="forecast")
    if lower is not None and upper is not None:
        ax.fill_between(
            x,
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            alpha=0.2,
            label="interval",
        )
    ax.set(xlabel="date", ylabel="value")
    ax.legend(loc="best")


@chart(title="Rolling mean & std")
def rolling_stats(ax: Axes, series: ArrayLike, *, window: int) -> None:
    """Series with a rolling-mean line and a +/-1 rolling-std band (trend & stationarity)."""
    values = pd.Series(np.asarray(series, dtype=float))
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    index = np.arange(values.size)
    ax.plot(index, values.to_numpy(), color="lightgrey", label="series")
    ax.plot(index, mean.to_numpy(), color="tab:blue", label=f"rolling mean ({window})")
    ax.fill_between(index, (mean - std).to_numpy(), (mean + std).to_numpy(), alpha=0.2)
    ax.set(xlabel="index", ylabel="value")
    ax.legend(loc="best")


@chart(title="Lag plot")
def lag_plot(ax: Axes, series: ArrayLike, *, lag: int = 1) -> None:
    """Scatter of each value against the value ``lag`` steps earlier."""
    values = np.asarray(series, dtype=float)
    ax.scatter(values[:-lag], values[lag:], alpha=0.4, edgecolor="none")
    ax.set(xlabel="value(t)", ylabel=f"value(t + {lag})")


@chart(title="Seasonal subseries")
def seasonal_subseries(ax: Axes, series: ArrayLike, *, period: int) -> None:
    """Overlay each seasonal cycle plus the mean profile across the period."""
    values = np.asarray(series, dtype=float)
    n_cycles = values.size // period
    cycles = values[: n_cycles * period].reshape(n_cycles, period)
    positions = np.arange(period)
    for cycle in cycles:
        ax.plot(positions, cycle, color="lightgrey", linewidth=0.8)
    ax.plot(positions, cycles.mean(axis=0), color="tab:blue", linewidth=2, label="mean")
    ax.set(xlabel="position within period", ylabel="value")
    ax.legend(loc="best")


@chart(title="Forecast residuals")
def forecast_residuals(ax: Axes, residuals: ArrayLike, *, dates: ArrayLike | None = None) -> None:
    """Forecast errors over time; they should hover around zero with no trend."""
    resid = np.asarray(residuals, dtype=float)
    x = np.arange(resid.size) if dates is None else np.asarray(dates)
    ax.plot(x, resid, marker=".", linestyle="-")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(xlabel="time" if dates is not None else "index", ylabel="residual")


@chart(title="Autocorrelation (ACF)")
def acf(ax: Axes, series: ArrayLike, *, lags: int = 40) -> None:
    """Autocorrelation function up to ``lags``."""
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(np.asarray(series, dtype=float), lags=lags, ax=ax)


@chart(title="Partial autocorrelation (PACF)")
def pacf(ax: Axes, series: ArrayLike, *, lags: int = 40) -> None:
    """Partial autocorrelation function up to ``lags``."""
    from statsmodels.graphics.tsaplots import plot_pacf

    plot_pacf(np.asarray(series, dtype=float), lags=lags, ax=ax)


@chart(title="Cohort retention")
def cohort_heatmap(ax: Axes, retention: pl.DataFrame, *, index: str) -> None:
    """Retention heatmap from a cohort x period table; ``index`` names the cohort column."""
    pdf = retention.to_pandas().set_index(index)
    sns.heatmap(pdf, annot=True, fmt=".0%", cmap="Blues", ax=ax)
    ax.set(xlabel="period", ylabel=index)


def seasonal_decomposition(values: ArrayLike, *, period: int, model: str = "additive") -> Figure:
    """Trend/seasonal/residual decomposition (statsmodels). Multi-panel, so it returns a Figure."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(np.asarray(values, dtype=float), period=period, model=model)
    figure = cast(Figure, result.plot())
    figure.set_size_inches(10, 8)
    return figure


@chart(title="Survival curve")
def survival_curve(
    ax: Axes,
    curves: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    time: str = "time",
    survival: str = "survival",
    ci_low: str = "ci_low",
    ci_high: str = "ci_high",
) -> None:
    """Kaplan-Meier step curve(s) with CI bands; pass ``{label: frame}`` to compare segments.

    Takes ``modeling.survival.kaplan_meier`` output. Survival is a step function (it drops at
    event times and is flat between), so steps — not smooth lines — are the honest drawing.
    """
    groups = curves if isinstance(curves, dict) else {"": curves}
    for label, frame in groups.items():
        times = np.concatenate([[0.0], frame[time].to_numpy()])
        values = np.concatenate([[1.0], frame[survival].to_numpy()])
        (line,) = ax.step(times, values, where="post", label=label or None)
        if ci_low in frame.columns and ci_high in frame.columns:
            ax.fill_between(
                frame[time].to_numpy(),
                frame[ci_low].to_numpy(),
                frame[ci_high].to_numpy(),
                step="post",
                alpha=0.15,
                color=line.get_color(),
            )
    if len(groups) > 1:
        ax.legend()
    ax.set(xlabel="time", ylabel="survival probability", ylim=(0.0, 1.05))
