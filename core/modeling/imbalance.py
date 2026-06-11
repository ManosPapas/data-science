"""Handling rare positives (fraud, churn): resampling, class weights, threshold tuning.

Resampling needs imbalanced-learn (the ``imbalance`` extra); class weights and threshold tuning are
pure sklearn/numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight


def make_resampler(strategy: str = "smote", *, seed: int = 42, **params: Any) -> Any:
    """Build a resampler: smote / random_over / random_under / smote_tomek / smoteenn."""
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=seed, **params)
    if strategy == "random_over":
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=seed, **params)
    if strategy == "random_under":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=seed, **params)
    if strategy == "smote_tomek":
        from imblearn.combine import SMOTETomek

        return SMOTETomek(random_state=seed, **params)
    if strategy == "smoteenn":
        from imblearn.combine import SMOTEENN

        return SMOTEENN(random_state=seed, **params)
    raise ValueError(f"unknown resampler strategy: {strategy}")


def imbalanced_pipeline(model: Any, *, resampler: Any, preprocessor: Any | None = None) -> Any:
    """imblearn Pipeline: (preprocessor ->) resampler -> model. Resamples train folds only."""
    from imblearn.pipeline import Pipeline as ImbPipeline

    steps: list[Any] = []
    if preprocessor is not None:
        steps.append(("pre", preprocessor))
    steps.append(("resample", resampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


def class_weights(y: ArrayLike) -> dict[Any, float]:
    """Balanced class weights {class: weight} for ``class_weight=`` / ``scale_pos_weight``."""
    target = np.asarray(y)
    classes = np.unique(target)
    weights = compute_class_weight("balanced", classes=classes, y=target)
    return dict(zip(classes.tolist(), weights.tolist(), strict=True))


def tune_threshold(y_true: ArrayLike, y_score: ArrayLike, *, metric: str = "f1") -> float:
    """Decision threshold on ``y_score`` maximizing the metric (f1 / precision / recall)."""
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    scorers = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    scorer = scorers[metric]
    thresholds = np.linspace(0.05, 0.95, 19)
    values = [scorer(truth, (scores >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(values))])
