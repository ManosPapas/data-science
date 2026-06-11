"""Forecasting models behind one interface — ``fit(y[, x])`` then ``predict(horizon[, x])``.

Baselines (naive / seasonal_naive / mean), classical statsmodels (arima/sarimax, ets/holt_winters),
and an ``ml`` reduction that forecasts recursively with any sklearn regressor over lagged targets.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _Baseline:
    """Naive (last value), seasonal-naive (last season), or mean forecast."""

    def __init__(self, *, strategy: str = "naive", season_length: int = 1) -> None:
        self.strategy = strategy
        self.season_length = season_length
        self._y: NDArray[np.float64] = np.empty(0)

    def fit(self, y: ArrayLike, x: Any = None) -> Self:
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, horizon: int, x: Any = None) -> NDArray[np.float64]:
        if self.strategy == "mean":
            return np.full(horizon, float(self._y.mean()))
        if self.strategy == "seasonal_naive":
            season = self._y[-self.season_length :]
            repeats = int(np.ceil(horizon / self.season_length))
            return np.asarray(np.tile(season, repeats)[:horizon], dtype=float)
        return np.full(horizon, float(self._y[-1]))


class _Classical:
    """statsmodels SARIMAX (arima/sarimax) or ExponentialSmoothing (ets/holt_winters)."""

    def __init__(self, kind: str, **params: Any) -> None:
        self.kind = kind
        self.params = params
        self._result: Any = None

    def fit(self, y: ArrayLike, x: ArrayLike | None = None) -> Self:
        series = np.asarray(y, dtype=float)
        if self.kind == "ets":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._result = ExponentialSmoothing(series, **self.params).fit()
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            exog = np.asarray(x, dtype=float) if x is not None else None
            self._result = SARIMAX(series, exog=exog, **self.params).fit(disp=False)
        return self

    def predict(self, horizon: int, x: ArrayLike | None = None) -> NDArray[np.float64]:
        if self.kind == "ets":
            return np.asarray(self._result.forecast(horizon), dtype=float)
        exog = np.asarray(x, dtype=float) if x is not None else None
        return np.asarray(self._result.forecast(horizon, exog=exog), dtype=float)


class _MLReduction:
    """Recursive lag-feature forecasting with any sklearn regressor."""

    def __init__(self, estimator: Any, *, lags: int = 7) -> None:
        self.estimator = estimator
        self.lags = lags
        self._history: NDArray[np.float64] = np.empty(0)

    def fit(self, y: ArrayLike, x: Any = None) -> Self:
        self._history = np.asarray(y, dtype=float)
        rows = [self._history[i - self.lags : i] for i in range(self.lags, self._history.size)]
        self.estimator.fit(np.asarray(rows), self._history[self.lags :])
        return self

    def predict(self, horizon: int, x: Any = None) -> NDArray[np.float64]:
        history = list(self._history)
        forecasts: list[float] = []
        for _ in range(horizon):
            window = np.asarray(history[-self.lags :]).reshape(1, -1)
            value = float(self.estimator.predict(window)[0])
            forecasts.append(value)
            history.append(value)
        return np.asarray(forecasts, dtype=float)


_BASELINES = {"naive", "seasonal_naive", "mean"}
_CLASSICAL = {"arima": "sarimax", "sarimax": "sarimax", "ets": "ets", "holt_winters": "ets"}


def make_forecaster(name: str = "naive", **params: Any) -> Any:
    """Build a forecaster with a uniform ``fit(y)`` / ``predict(horizon)``.

    Names: naive, seasonal_naive, mean, arima, sarimax, ets/holt_winters, ml. ``ml`` needs
    ``estimator=`` (any sklearn regressor, e.g. from ``registry.make_model``).
    """
    if name in _BASELINES:
        return _Baseline(strategy=name, season_length=params.get("season_length", 1))
    if name in _CLASSICAL:
        return _Classical(_CLASSICAL[name], **params)
    if name == "ml":
        estimator = params.pop("estimator")
        return _MLReduction(estimator, lags=params.pop("lags", 7))
    raise ValueError(f"unknown forecaster: {name}")
