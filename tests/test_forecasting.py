"""Tests for forecasting models, prediction intervals, and backtesting."""

from __future__ import annotations

import numpy as np

from core.forecasting import backtest
from core.forecasting.models import make_forecaster


def _series(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(120)
    return 50.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, t.size)


def test_baseline_forecasters_shape(rng: np.random.Generator) -> None:
    y = _series(rng)
    for name, kw in [("naive", {}), ("mean", {}), ("seasonal_naive", {"season_length": 12})]:
        forecaster = make_forecaster(name, **kw)
        forecaster.fit(y)
        assert forecaster.predict(6).shape == (6,)


def test_ets_and_arima_predict(rng: np.random.Generator) -> None:
    y = _series(rng)
    ets = make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=12)
    arima = make_forecaster("arima", order=(1, 1, 1))
    assert ets.fit(y).predict(6).shape == (6,)
    assert arima.fit(y).predict(6).shape == (6,)


def test_prediction_interval_brackets_point(rng: np.random.Generator) -> None:
    y = _series(rng)
    forecaster = make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=12)
    forecaster.fit(y)
    point = forecaster.predict(6)
    lower, upper = forecaster.predict_interval(6)
    assert lower.shape == upper.shape == (6,)
    assert np.all(lower <= point + 1e-6)
    assert np.all(point <= upper + 1e-6)
    width = upper - lower
    assert np.all(width > 0)  # a real, non-degenerate band
    assert width[-1] > width[0]  # uncertainty grows with the horizon


def test_ml_reduction_forecaster(rng: np.random.Generator) -> None:
    from core.modeling.registry import make_model

    y = _series(rng)
    estimator = make_model("random_forest", task="regression", n_estimators=30, random_state=0)
    forecaster = make_forecaster("ml", estimator=estimator, lags=12)
    forecaster.fit(y)
    assert forecaster.predict(6).shape == (6,)


def test_backtest_metrics() -> None:
    truth = [10.0, 12.0, 11.0]
    pred = [11.0, 11.0, 11.0]
    assert backtest.mae(truth, pred) >= 0
    assert backtest.rmse(truth, pred) >= 0
    assert backtest.mape(truth, pred) >= 0
    assert backtest.smape(truth, pred) >= 0


def test_rolling_origin(rng: np.random.Generator) -> None:
    y = _series(rng)
    result = backtest.rolling_origin(
        lambda: make_forecaster("naive"), y, initial=80, horizon=6, step=12
    )
    assert {"origin", "h", "actual", "pred"} <= set(result.columns)
    assert result.height > 0
