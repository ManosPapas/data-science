"""Anomaly detection — unsupervised outlier scoring (fraud, quality, monitoring)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.modeling.frames import to_features


def make_detector(name: str = "isolation_forest", **params: Any) -> Any:
    """Build a detector: isolation_forest / local_outlier_factor / one_class_svm."""
    if name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(**params)
    if name == "local_outlier_factor":
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(**params)
    if name == "one_class_svm":
        from sklearn.svm import OneClassSVM

        return OneClassSVM(**params)
    raise ValueError(f"unknown detector: {name}")


def anomaly_labels(detector: Any, x: Any) -> NDArray[np.int_]:
    """Fit-predict outlier labels (1 = inlier, -1 = outlier)."""
    return np.asarray(detector.fit_predict(to_features(x)))
