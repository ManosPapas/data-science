"""Forecasting models behind one interface — ``fit(y[, x])`` then ``predict(horizon[, x])``.

Baselines (naive / seasonal_naive / mean), classical statsmodels (arima/sarimax, ets/holt_winters),
and an ``ml`` reduction that forecasts recursively with any sklearn regressor over lagged targets.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normal_band(
    point: NDArray[np.float64], sigma: float, alpha: float, *, grow: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric normal prediction band around ``point``; widens with horizon when ``grow``."""
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    steps = np.sqrt(np.arange(1, point.size + 1)) if grow else np.ones(point.size)
    half = z * sigma * steps
    return point - half, point + half


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

    def predict_interval(
        self, horizon: int, *, alpha: float = 0.05, x: Any = None, point: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        center = np.asarray(point, dtype=float) if point is not None else self.predict(horizon)
        if self.strategy == "mean":
            sigma_mean = float(self._y.std(ddof=1)) if self._y.size > 1 else 0.0
            return _normal_band(center, sigma_mean, alpha, grow=False)
        if self.strategy == "seasonal_naive":
            diffs = self._y[self.season_length :] - self._y[: -self.season_length]
        else:
            diffs = np.diff(self._y)
        sigma = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
        return _normal_band(center, sigma, alpha, grow=True)


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

    def predict_interval(
        self,
        horizon: int,
        *,
        alpha: float = 0.05,
        x: ArrayLike | None = None,
        point: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Pass an already-computed ``point`` forecast to skip recomputing it (ETS path)."""
        if self.kind == "ets":
            center = np.asarray(point, dtype=float) if point is not None else self.predict(horizon)
            sigma = float(np.std(np.asarray(self._result.resid, dtype=float), ddof=1))
            return _normal_band(center, sigma, alpha, grow=True)
        exog = np.asarray(x, dtype=float) if x is not None else None
        forecast = self._result.get_forecast(horizon, exog=exog)
        ci = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
        return ci[:, 0], ci[:, 1]


class _MLReduction:
    """Recursive lag-feature forecasting with any sklearn regressor."""

    def __init__(self, estimator: Any, *, lags: int = 7) -> None:
        self.estimator = estimator
        self.lags = lags
        self._history: NDArray[np.float64] = np.empty(0)
        self._sigma: float = 0.0  # one-step holdout residual std, set at fit time

    def fit(self, y: ArrayLike, x: Any = None) -> Self:
        self._history = np.asarray(y, dtype=float)
        rows = np.asarray(
            [self._history[i - self.lags : i] for i in range(self.lags, self._history.size)]
        )
        target = self._history[self.lags :]
        self.estimator.fit(rows, target)
        self._sigma = self._holdout_sigma(rows, target)
        return self

    def _holdout_sigma(self, rows: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        """One-step residual std on a time-ordered holdout.

        In-sample residuals of low-bias learners (forests, boosters) are near zero, which would
        make the band dishonestly narrow — so probe with a clone fit on the first 75% and measure
        on the rest; fall back to in-sample only when the series is too short.
        """
        from sklearn.base import clone

        cut = int(target.size * 0.75)
        if target.size - cut >= 3:
            probe = clone(self.estimator).fit(rows[:cut], target[:cut])
            resid = target[cut:] - np.asarray(probe.predict(rows[cut:]), dtype=float)
        else:
            resid = target - np.asarray(self.estimator.predict(rows), dtype=float)
        return float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

    def predict(self, horizon: int, x: Any = None) -> NDArray[np.float64]:
        history = list(self._history)
        forecasts: list[float] = []
        for _ in range(horizon):
            window = np.asarray(history[-self.lags :]).reshape(1, -1)
            value = float(self.estimator.predict(window)[0])
            forecasts.append(value)
            history.append(value)
        return np.asarray(forecasts, dtype=float)

    def predict_interval(
        self, horizon: int, *, alpha: float = 0.05, x: Any = None, point: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # sigma is the one-step holdout residual std cached at fit time; the band widens with
        # sqrt(horizon) — an approximation for the recursive multi-step forecast.
        center = np.asarray(point, dtype=float) if point is not None else self.predict(horizon)
        return _normal_band(center, self._sigma, alpha, grow=True)


_BASELINES = {"naive", "seasonal_naive", "mean"}
_CLASSICAL = {"arima": "sarimax", "sarimax": "sarimax", "ets": "ets", "holt_winters": "ets"}


def make_forecaster(name: str = "naive", **params: Any) -> Any:
    """Build a forecaster with a uniform ``fit(y)`` / ``predict(horizon)`` / ``predict_interval``.

    Names: naive, seasonal_naive, mean, arima, sarimax, ets/holt_winters, ml. ``ml`` needs
    ``estimator=`` (any sklearn regressor, e.g. from ``registry.make_model``).
    ``predict_interval(horizon, point=...)`` reuses an already-computed point forecast.
    """
    if name in _BASELINES:
        return _Baseline(strategy=name, season_length=params.get("season_length", 1))
    if name in _CLASSICAL:
        return _Classical(_CLASSICAL[name], **params)
    if name == "ml":
        estimator = params.pop("estimator")
        return _MLReduction(estimator, lags=params.pop("lags", 7))
    raise ValueError(f"unknown forecaster: {name}")
