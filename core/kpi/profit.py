"""Profit / cost KPIs — turn binary predictions into money via a cost/benefit matrix.

``costs`` maps per-cell values, e.g. for fraud:
``{"tp": 0, "fp": -10, "tn": 0, "fn": -500}`` (a missed fraud costs far more than an investigation).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def expected_value(y_true: ArrayLike, y_pred: ArrayLike, *, costs: Mapping[str, float]) -> float:
    """Net value of binary predictions given per-cell values {tp, fp, tn, fn}."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return float(
        tp * costs.get("tp", 0.0)
        + fp * costs.get("fp", 0.0)
        + tn * costs.get("tn", 0.0)
        + fn * costs.get("fn", 0.0)
    )


def profit_curve(
    y_true: ArrayLike, y_score: ArrayLike, *, costs: Mapping[str, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expected value across decision thresholds; returns (thresholds, values)."""
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.0, 1.0, 101)
    values = np.array([expected_value(yt, (ys >= t).astype(int), costs=costs) for t in thresholds])
    return thresholds, values


def profit_threshold(
    y_true: ArrayLike, y_score: ArrayLike, *, costs: Mapping[str, float]
) -> tuple[float, float]:
    """Decision threshold that maximizes expected value; returns (threshold, value)."""
    thresholds, values = profit_curve(y_true, y_score, costs=costs)
    best = int(np.argmax(values))
    return float(thresholds[best]), float(values[best])
