"""Model metrics and diagnostics-compute — paired with the viz.model / viz.explain charts.

Metric helpers return dicts/frames; the ``*_scores`` functions do the heavy refitting for the
learning-curve, validation-curve, and feature-selection charts (compute here, plot in viz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from sklearn import metrics

from core.modeling.frames import to_features, to_target


def _feature_names(features: Any, n_features: int) -> list[str]:
    if hasattr(features, "columns"):
        return [str(name) for name in features.columns]
    return [f"x{i}" for i in range(n_features)]


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
    # Binary problems get the confusion-derived rates by name: recall above IS sensitivity (TPR);
    # specificity (TNR) and NPV are the negative-class mirrors a medical/fraud reviewer asks for.
    classes = np.unique(yt)
    if classes.size == 2:
        tn, fp, fn, tp = metrics.confusion_matrix(yt, yp, labels=classes).ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
        out["npv"] = float(tn / (tn + fn)) if (tn + fn) else 0.0
    # Score-based metrics are undefined when the slice has only one class (a CV fold or segment
    # that happens to be all-positive); skip them rather than let roc_auc_score crash the dict.
    if y_score is not None and classes.size == 2:
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


# --- Diagnostics compute (feeds viz.model / viz.explain) ----------------------------------------


def permutation_importance(
    model: Any,
    x: Any,
    y: Any,
    *,
    n_repeats: int = 10,
    scoring: Any = None,
    seed: int = 42,
    n_jobs: int = 1,
) -> pl.DataFrame:
    """Model-agnostic importance: how much the score drops when one feature is shuffled.

    Run on a *fitted* model with *held-out* data (on train it just mirrors overfitting). Unlike
    impurity importances it isn't biased toward high-cardinality features, but correlated features
    share credit — read clusters together. Returns (feature, importance_mean, importance_std)
    sorted, ready for ``viz.explain.permutation_importance``.
    """
    from sklearn.inspection import permutation_importance as _sk_permutation

    features = to_features(x)
    result = _sk_permutation(
        model,
        features,
        to_target(y),
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=seed,
        n_jobs=n_jobs,
    )
    names = _feature_names(features, np.asarray(features).shape[1])
    frame = pl.DataFrame(
        {
            "feature": names,
            "importance_mean": np.asarray(result.importances_mean, dtype=float),
            "importance_std": np.asarray(result.importances_std, dtype=float),
        }
    )
    return frame.sort("importance_mean", descending=True)


def learning_curve_scores(
    model: Any, x: Any, y: Any, *, cv: Any = 5, scoring: Any = None, sizes: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Train vs validation score as the training set grows — the bias/variance diagnostic.

    Both curves converging low = high bias (underfitting: add features/complexity); a persistent
    train-validation gap = high variance (overfitting: regularize, simplify, or get more data —
    the curve shows whether more data would even help). Returns (train_sizes, train_scores,
    val_scores) for ``viz.model.learning_curve``.
    """
    from sklearn.model_selection import learning_curve as _sk_learning_curve

    train_sizes, train_scores, val_scores = _sk_learning_curve(
        model,
        to_features(x),
        to_target(y),
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5) if sizes is None else np.asarray(sizes, dtype=float),
    )
    return (
        np.asarray(train_sizes, dtype=float),
        np.asarray(train_scores, dtype=float),
        np.asarray(val_scores, dtype=float),
    )


def validation_curve_scores(
    model: Any,
    x: Any,
    y: Any,
    *,
    param_name: str,
    param_range: ArrayLike,
    cv: Any = 5,
    scoring: Any = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Train vs validation score across one hyper-parameter — where overfitting starts.

    The validation curve peaks at the complexity that generalizes best; past it, train keeps
    rising while validation falls. Returns (param_values, train_scores, val_scores) for
    ``viz.model.validation_curve``.
    """
    from sklearn.model_selection import validation_curve as _sk_validation_curve

    train_scores, val_scores = _sk_validation_curve(
        model,
        to_features(x),
        to_target(y),
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
    )
    return (
        np.asarray(param_range, dtype=float),
        np.asarray(train_scores, dtype=float),
        np.asarray(val_scores, dtype=float),
    )


@dataclass(frozen=True)
class FeatureSelectionResult:
    """RFECV output: the CV curve (for the chart) plus the chosen feature subset."""

    n_features: NDArray[np.int_]
    mean_scores: NDArray[np.float64]
    std_scores: NDArray[np.float64]
    selected: list[str]


def rfecv_scores(
    model: Any,
    x: Any,
    y: Any,
    *,
    cv: Any = 5,
    scoring: Any = None,
    step: int = 1,
    min_features: int = 1,
) -> FeatureSelectionResult:
    """Recursive feature elimination with CV — how many (and which) features earn their keep.

    Drops the weakest feature(s) by ``coef_``/``feature_importances_``, re-scores, repeats; fewer
    features cut variance and ease interpretation at a possible cost in bias. Feed ``n_features`` /
    ``mean_scores`` / ``std_scores`` to ``viz.model.feature_selection_curve``; refit on
    ``selected``.
    """
    from sklearn.feature_selection import RFECV

    features = to_features(x)
    search = RFECV(model, step=step, cv=cv, scoring=scoring, min_features_to_select=min_features)
    search.fit(features, to_target(y))
    names = _feature_names(features, int(search.n_features_in_))
    selected = [name for name, keep in zip(names, search.support_, strict=True) if keep]
    return FeatureSelectionResult(
        np.asarray(search.cv_results_["n_features"], dtype=int),
        np.asarray(search.cv_results_["mean_test_score"], dtype=float),
        np.asarray(search.cv_results_["std_test_score"], dtype=float),
        selected,
    )


def ranking_metrics(relevance: ArrayLike, scores: ArrayLike, *, k: int = 10) -> dict[str, float]:
    """Ranking quality at cut-off ``k``: NDCG, precision@k, recall@k, MRR (means over queries).

    Inputs are (queries x items) matrices — one row per user/query — of true relevance (binary
    or graded) and model scores; 1-D inputs are treated as a single query. Classification
    metrics ignore *order*; these score what a recommender/search ranking actually serves.
    Rows without any relevant item are skipped (nothing to find).
    """
    rel = np.atleast_2d(np.asarray(relevance, dtype=float))
    sco = np.atleast_2d(np.asarray(scores, dtype=float))
    if rel.shape != sco.shape:
        raise ValueError("relevance and scores must have the same shape")
    if k < 1:
        raise ValueError("k must be at least 1")
    ndcg, precision, recall, mrr, kept = [], [], [], [], 0
    discounts = 1.0 / np.log2(np.arange(2, rel.shape[1] + 2))
    for row_rel, row_sco in zip(rel, sco, strict=True):
        total_relevant = float((row_rel > 0).sum())
        if total_relevant == 0:
            continue
        kept += 1
        order = np.argsort(row_sco)[::-1]
        ranked = row_rel[order]
        ideal = np.sort(row_rel)[::-1]
        top = min(k, ranked.size)
        dcg = float(np.sum(ranked[:top] * discounts[:top]))
        idcg = float(np.sum(ideal[:top] * discounts[:top]))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        hits = float((ranked[:top] > 0).sum())
        precision.append(hits / k)
        recall.append(hits / total_relevant)
        first = np.flatnonzero(ranked > 0)
        mrr.append(1.0 / (first[0] + 1.0) if first.size else 0.0)
    if kept == 0:
        raise ValueError("no query had a relevant item — ranking quality is undefined")
    return {
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        "mrr": float(np.mean(mrr)),
        "queries": float(kept),
    }
