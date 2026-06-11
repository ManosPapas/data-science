"""Rolling-origin backtesting and forecast error metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error (%)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric mean absolute percentage error (%)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100)


def rolling_origin(
    make: Callable[[], Any], y: ArrayLike, *, initial: int, horizon: int, step: int = 1
) -> pl.DataFrame:
    """Expanding-window backtest: grow train, forecast ``horizon`` ahead, record pred vs actual.

    ``make`` returns a fresh forecaster each fold, e.g. ``lambda: make_forecaster('ets')``.
    """
    series = np.asarray(y, dtype=float)
    rows: list[dict[str, Any]] = []
    origin = initial
    while origin + horizon <= series.size:
        forecaster = make()
        forecaster.fit(series[:origin])
        preds = np.asarray(forecaster.predict(horizon), dtype=float)
        actual = series[origin : origin + horizon]
        for step_ahead in range(horizon):
            rows.append(
                {
                    "origin": origin,
                    "h": step_ahead + 1,
                    "actual": float(actual[step_ahead]),
                    "pred": float(preds[step_ahead]),
                }
            )
        origin += step
    return pl.DataFrame(rows)
