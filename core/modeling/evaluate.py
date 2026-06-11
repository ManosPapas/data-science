"""Model metrics — dicts and a per-class report, paired with the viz.model charts."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn import metrics


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float | None]:
    """RMSE, MAE, median-AE, max-error, MAPE, RMSLE (if non-negative), R2, explained variance."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    can_log = bool((yt >= 0).all() and (yp >= 0).all())
    return {
        "rmse": float(metrics.root_mean_squared_error(yt, yp)),
        "mae": float(metrics.mean_absolute_error(yt, yp)),
        "median_ae": float(metrics.median_absolute_error(yt, yp)),
        "max_error": float(metrics.max_error(yt, yp)),
        "mape": float(metrics.mean_absolute_percentage_error(yt, yp)),
        "rmsle": float(metrics.root_mean_squared_log_error(yt, yp)) if can_log else None,
        "r2": float(metrics.r2_score(yt, yp)),
        "explained_variance": float(metrics.explained_variance_score(yt, yp)),
    }


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, *, y_score: ArrayLike | None = None
) -> dict[str, float]:
    """Accuracy, balanced-acc, weighted P/R/F1, MCC, kappa; AUC/AP/log-loss/Brier need scores."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out = {
        "accuracy": float(metrics.accuracy_score(yt, yp)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(yt, yp)),
        "precision": float(metrics.precision_score(yt, yp, average="weighted", zero_division=0)),
        "recall": float(metrics.recall_score(yt, yp, average="weighted", zero_division=0)),
        "f1": float(metrics.f1_score(yt, yp, average="weighted", zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(yt, yp)),
        "cohen_kappa": float(metrics.cohen_kappa_score(yt, yp)),
    }
    if y_score is not None:
        scores = np.asarray(y_score, dtype=float)
        out["roc_auc"] = float(metrics.roc_auc_score(yt, scores))
        out["average_precision"] = float(metrics.average_precision_score(yt, scores))
        out["log_loss"] = float(metrics.log_loss(yt, scores))
        out["brier"] = float(metrics.brier_score_loss(yt, scores))
    return out


def report(y_true: ArrayLike, y_pred: ArrayLike) -> pl.DataFrame:
    """Per-class precision / recall / f1 / support as a Polars frame."""
    raw = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows: list[dict[str, Any]] = []
    for label, row in raw.items():
        if isinstance(row, dict):
            rows.append(
                {
                    "label": str(label),
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": row["support"],
                }
            )
    return pl.DataFrame(rows)


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, *, alpha: float = 0.5) -> float:
    """Pinball (quantile) loss at quantile ``alpha`` — for quantile regression / intervals."""
    return float(
        metrics.mean_pinball_loss(
            np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), alpha=alpha
        )
    )
