"""Model-evaluation charts: classification, regression, and learning/selection diagnostics.

Lightweight metric computation (ROC/PR points, confusion) happens inside the chart from the vectors
you pass. Heavier work (refitting across sizes/features for learning or selection curves) is done in
modeling and the resulting score arrays are passed in.

For a regression residual histogram or Q-Q, use ``eda.histogram``/``eda.qq`` on ``y_true - y_pred``;
for a classifier KS, use ``eda.ks`` on the per-class score arrays — no duplicate charts here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from core.viz.base import chart

# --- Classification ---------------------------------------------------------------------------


@chart(title="ROC curve")
def roc(ax: Axes, y_true: ArrayLike, y_score: ArrayLike) -> None:
    """Receiver-operating-characteristic curve with the AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right")


@chart(title="Precision-recall curve")
def precision_recall(ax: Axes, y_true: ArrayLike, y_score: ArrayLike) -> None:
    """Precision-recall curve with the average precision in the legend."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ax.plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_score):.3f}")
    ax.set(xlabel="recall", ylabel="precision")
    ax.legend(loc="lower left")


@chart(title="Metrics vs threshold")
def threshold_curve(ax: Axes, y_true: ArrayLike, y_score: ArrayLike) -> None:
    """Precision, recall, and F1 vs the decision threshold (to choose an operating point)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, f1, label="F1")
    ax.set(xlabel="threshold", ylabel="score")
    ax.legend(loc="best")


@chart(title="Score distribution by class")
def score_distribution(ax: Axes, y_true: ArrayLike, y_score: ArrayLike, *, bins: int = 30) -> None:
    """Predicted-score histograms split by actual class — visual class separation."""
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    for label in np.unique(truth):
        ax.hist(scores[truth == label], bins=bins, alpha=0.5, label=f"class {label}")
    ax.set(xlabel="predicted score", ylabel="count")
    ax.legend(loc="best")


@chart(title="Confusion matrix")
def confusion(
    ax: Axes, y_true: ArrayLike, y_pred: ArrayLike, *, normalize: str | None = None
) -> None:
    """Confusion matrix; pass ``normalize='true'`` for row-normalized rates."""
    matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
    ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)


@chart(title="Calibration curve")
def calibration(ax: Axes, y_true: ArrayLike, y_prob: ArrayLike, *, n_bins: int = 10) -> None:
    """Reliability curve: observed frequency vs predicted probability."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set(xlabel="mean predicted probability", ylabel="fraction of positives")
    ax.legend(loc="upper left")


@chart(title="Cumulative gains")
def gains_curve(ax: Axes, y_true: ArrayLike, y_score: ArrayLike) -> None:
    """Cumulative share of positives captured as the population is ranked by score."""
    truth = np.asarray(y_true, dtype=float)
    order = np.argsort(np.asarray(y_score, dtype=float))[::-1]
    gains = np.cumsum(truth[order]) / truth.sum()
    population = np.arange(1, truth.size + 1) / truth.size
    ax.plot(population, gains, label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set(xlabel="proportion of samples", ylabel="proportion of positives captured")
    ax.legend(loc="lower right")


@chart(title="Lift by decile")
def lift_curve(ax: Axes, y_true: ArrayLike, y_score: ArrayLike, *, n_bins: int = 10) -> None:
    """Lift over the base rate for each score decile (highest scores first)."""
    truth = np.asarray(y_true, dtype=float)
    ranked = truth[np.argsort(np.asarray(y_score, dtype=float))[::-1]]
    base_rate = truth.mean()
    lift = [seg.mean() / base_rate if base_rate else 0.0 for seg in np.array_split(ranked, n_bins)]
    ax.bar(np.arange(1, n_bins + 1), lift, color="tab:blue")
    ax.axhline(1.0, linestyle="--", color="grey")
    ax.set(xlabel="decile (1 = highest score)", ylabel="lift")


@chart(title="Classification report")
def classification_report_heatmap(ax: Axes, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Precision / recall / F1 per class as a heatmap."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).transpose().drop(index="accuracy", errors="ignore")
    sns.heatmap(
        frame[["precision", "recall", "f1-score"]], annot=True, fmt=".2f", cmap="Blues", ax=ax
    )


# --- Regression -------------------------------------------------------------------------------


@chart(title="Predicted vs actual")
def predicted_vs_actual(ax: Axes, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Scatter of predictions against truth with the 45-degree reference line."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    ax.scatter(actual, predicted, alpha=0.4, edgecolor="none")
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="grey")
    ax.set(xlabel="actual", ylabel="predicted")


@chart(title="Residuals vs predicted")
def residuals(ax: Axes, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Residuals against predictions; look for structure (it should be a flat cloud)."""
    predicted = np.asarray(y_pred, dtype=float)
    resid = np.asarray(y_true, dtype=float) - predicted
    ax.scatter(predicted, resid, alpha=0.4, edgecolor="none")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(xlabel="predicted", ylabel="residual")


@chart(title="Scale-location")
def scale_location(ax: Axes, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """sqrt(|standardized residual|) vs fitted — diagnoses non-constant variance."""
    predicted = np.asarray(y_pred, dtype=float)
    resid = np.asarray(y_true, dtype=float) - predicted
    std = resid.std()
    standardized = resid / std if std else resid
    ax.scatter(predicted, np.sqrt(np.abs(standardized)), alpha=0.4, edgecolor="none")
    ax.set(xlabel="predicted", ylabel="sqrt(|standardized residual|)")


@chart(title="Residuals vs leverage")
def residuals_vs_leverage(
    ax: Axes,
    leverage: ArrayLike,
    standardized_residuals: ArrayLike,
    *,
    cooks_distance: ArrayLike | None = None,
) -> None:
    """Influence diagnostic. Pass precomputed leverage + standardized residuals (e.g. statsmodels
    ``OLSInfluence``); point size encodes Cook's distance when provided."""
    lev = np.asarray(leverage, dtype=float)
    std_resid = np.asarray(standardized_residuals, dtype=float)
    sizes = None if cooks_distance is None else 800.0 * np.asarray(cooks_distance, dtype=float)
    ax.scatter(lev, std_resid, s=sizes, alpha=0.5, edgecolor="none")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(xlabel="leverage", ylabel="standardized residual")


@chart(title="Error by feature")
def error_by_feature(ax: Axes, feature: ArrayLike, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Residual against a feature's value — reveals where the model is biased."""
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    ax.scatter(np.asarray(feature, dtype=float), resid, alpha=0.4, edgecolor="none")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(xlabel="feature value", ylabel="residual")


@chart(title="Regression calibration")
def regression_calibration(
    ax: Axes, y_true: ArrayLike, y_pred: ArrayLike, *, n_bins: int = 10
) -> None:
    """Binned mean-actual vs mean-predicted — are predictions right on average per band?"""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    order = np.argsort(predicted)
    mean_pred = [seg.mean() for seg in np.array_split(predicted[order], n_bins)]
    mean_actual = [seg.mean() for seg in np.array_split(actual[order], n_bins)]
    ax.plot(mean_pred, mean_actual, marker="o", label="model")
    lo, hi = float(predicted.min()), float(predicted.max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", label="ideal")
    ax.set(xlabel="mean predicted", ylabel="mean actual")
    ax.legend(loc="best")


# --- Learning & selection diagnostics ---------------------------------------------------------


@chart(title="Learning curve")
def learning_curve(
    ax: Axes, train_sizes: ArrayLike, train_scores: ArrayLike, val_scores: ArrayLike
) -> None:
    """Train vs validation score as training-set size grows (inputs from sklearn.learning_curve)."""
    sizes = np.asarray(train_sizes, dtype=float)
    _score_band(ax, sizes, train_scores, "train")
    _score_band(ax, sizes, val_scores, "validation")
    ax.set(xlabel="training examples", ylabel="score")
    ax.legend(loc="best")


@chart(title="Validation curve")
def validation_curve(
    ax: Axes, param_values: ArrayLike, train_scores: ArrayLike, val_scores: ArrayLike
) -> None:
    """Train vs validation score across a hyper-parameter range."""
    values = np.asarray(param_values, dtype=float)
    _score_band(ax, values, train_scores, "train")
    _score_band(ax, values, val_scores, "validation")
    ax.set(xlabel="parameter value", ylabel="score")
    ax.legend(loc="best")


@chart(title="Feature-selection curve")
def feature_selection_curve(
    ax: Axes, n_features: ArrayLike, scores: ArrayLike, *, std: ArrayLike | None = None
) -> None:
    """CV score vs number of features (RFECV-style) — the 'elbow' for how many features to keep.

    Marks the peak. Pass precomputed (n_features, mean CV score); optional ``std`` adds a band.
    """
    counts = np.asarray(n_features, dtype=float)
    mean = np.asarray(scores, dtype=float)
    ax.plot(counts, mean, marker="o")
    if std is not None:
        spread = np.asarray(std, dtype=float)
        ax.fill_between(counts, mean - spread, mean + spread, alpha=0.15)
    best = int(np.argmax(mean))
    ax.axvline(counts[best], linestyle="--", color="grey", label=f"best = {int(counts[best])}")
    ax.set(xlabel="number of features", ylabel="cross-validation score")
    ax.legend(loc="best")


def _score_band(ax: Axes, x: ArrayLike, scores: ArrayLike, label: str) -> None:
    """Plot the mean of a (n_points, n_folds) score matrix with a +/-1 std band."""
    matrix = np.asarray(scores, dtype=float)
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1)
    line = ax.plot(x, mean, marker="o", label=label)[0]
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())


@chart(title="Model comparison")
def model_comparison(ax: Axes, scores_by_model: dict[str, ArrayLike]) -> None:
    """Box plot of per-fold scores per model — overlap means no real difference."""
    labels = list(scores_by_model)
    data = [np.asarray(scores_by_model[name], dtype=float) for name in labels]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.set(ylabel="fold score")
