"""Fit / cross-validate / predict / partial-fit any model (optionally behind a preprocessor)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn import model_selection
from sklearn.pipeline import Pipeline

from core.modeling.frames import to_features, to_target


def fit(model: Any, x: Any, y: Any, *, preprocessor: Any | None = None) -> Any:
    """Fit ``model`` (behind ``preprocessor`` if given); return the fitted estimator."""
    if preprocessor is not None:
        estimator = Pipeline([("pre", preprocessor), ("model", model)])
    else:
        estimator = model
    estimator.fit(to_features(x), to_target(y))
    return estimator


def cross_validate(
    model: Any, x: Any, y: Any, *, cv: Any = 5, scoring: Any = None
) -> dict[str, Any]:
    """Cross-validation scores. Pass a splitter (see ``split.make_cv``) as ``cv``."""
    return dict(
        model_selection.cross_validate(model, to_features(x), to_target(y), cv=cv, scoring=scoring)
    )


def cross_val_predict(
    model: Any, x: Any, y: Any, *, cv: Any = 5, method: str = "predict"
) -> NDArray[Any]:
    """Out-of-fold predictions — each row predicted by a fold-model that never saw it.

    Use ``method='predict_proba'`` for probabilities (e.g. to feed the ROC / threshold tools).
    """
    predictions = model_selection.cross_val_predict(
        model, to_features(x), to_target(y), cv=cv, method=method
    )
    return np.asarray(predictions)


def predict(model: Any, x: Any) -> NDArray[Any]:
    """Predictions for a (possibly Polars) feature frame."""
    return np.asarray(model.predict(to_features(x)))


def predict_proba(model: Any, x: Any) -> NDArray[Any]:
    """Class probabilities for a (possibly Polars) feature frame."""
    return np.asarray(model.predict_proba(to_features(x)))


def score_frame(model: Any, df: pl.DataFrame, *, column: str = "prediction") -> pl.DataFrame:
    """Batch-score fresh data: return the feature frame with a prediction column appended."""
    return df.with_columns(pl.Series(column, model.predict(to_features(df))))


def score(model: Any, x: Any, *, positive_class: int = 1) -> NDArray[np.float64]:
    """A model's scalar score per row: class probability for classifiers, prediction otherwise.

    The single definition of "the number governance/explanation tools judge" — used by
    ``modeling.checks`` and ``modeling.interpret``. ``positive_class`` selects the probability
    column for classifiers (default the positive class of a binary model; set it for multiclass).
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(predict_proba(model, x)[:, positive_class], dtype=float)
    return np.asarray(predict(model, x), dtype=float)


def partial_fit(model: Any, x: Any, y: Any, *, classes: Any = None) -> Any:
    """Incrementally update a partial_fit model (sgd/mlp/naive bayes); pass ``classes`` once."""
    if classes is not None:
        model.partial_fit(to_features(x), to_target(y), classes=classes)
    else:
        model.partial_fit(to_features(x), to_target(y))
    return model
