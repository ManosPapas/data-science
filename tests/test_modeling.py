"""Tests for the modeling registry + train/evaluate harness."""

from __future__ import annotations

import numpy as np

from core.modeling import evaluate, registry, train


def test_make_model_returns_estimator() -> None:
    model = registry.make_model("ridge", task="regression", alpha=0.5)
    assert hasattr(model, "fit")


def test_available_models_nonempty() -> None:
    assert "xgboost" in registry.available_models("regression")


def test_fit_and_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    model = train.fit(registry.make_model("linear", task="regression"), x, y)
    predictions = train.predict(model, x)
    assert predictions.shape == (60,)


def test_regression_metrics() -> None:
    metrics = evaluate.regression_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert metrics["rmse"] >= 0.0
