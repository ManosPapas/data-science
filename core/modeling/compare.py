"""Compare models correctly: identical CV folds, mean & std per metric, and paired significance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn import model_selection

from core.modeling.frames import to_features, to_target


@dataclass(frozen=True)
class PairedResult:
    """Paired-comparison result: test statistic, p-value, and which model led ('a'/'b'/'tie')."""

    statistic: float
    p_value: float
    better: str


def fold_scores(
    model: Any, x: Any, y: Any, *, cv: Any = 5, scoring: str = "accuracy", n_jobs: int = 1
) -> NDArray[np.float64]:
    """Per-fold test scores for one model + metric (use the same ``cv`` across models).

    ``n_jobs`` fits the folds in parallel (``-1`` = all cores).
    """
    scores = model_selection.cross_val_score(
        model, to_features(x), to_target(y), cv=cv, scoring=scoring, n_jobs=n_jobs
    )
    return np.asarray(scores, dtype=float)


def leaderboard(
    models: Mapping[str, Any],
    x: Any,
    y: Any,
    *,
    cv: Any = 5,
    scoring: str | Sequence[str] = "accuracy",
    n_jobs: int = 1,
) -> pl.DataFrame:
    """Rank models on identical CV folds; returns mean & std per metric (sorted by the first).

    Pass a splitter from ``split.make_cv`` as ``cv`` so every model sees the same folds. ``n_jobs``
    parallelizes the fold fits within each model (``-1`` = all cores).
    """
    metrics = [scoring] if isinstance(scoring, str) else list(scoring)
    features, target = to_features(x), to_target(y)
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        result = model_selection.cross_validate(
            model, features, target, cv=cv, scoring=metrics, n_jobs=n_jobs
        )
        row: dict[str, Any] = {"model": name}
        for metric in metrics:
            scores = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(scores))
            row[f"{metric}_std"] = float(np.std(scores))
        rows.append(row)
    return pl.DataFrame(rows).sort(f"{metrics[0]}_mean", descending=True)


def paired_test(
    scores_a: ArrayLike, scores_b: ArrayLike, *, method: str = "t", alternative: str = "two-sided"
) -> PairedResult:
    """Paired significance test on per-fold scores (same folds) — is A really different from B?

    ``method='t'`` is a paired t-test, ``'wilcoxon'`` the non-parametric one. Folds aren't
    independent, so treat p-values as a guide.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if method == "wilcoxon":
        statistic, p_value = stats.wilcoxon(a, b, alternative=alternative)
    else:
        statistic, p_value = stats.ttest_rel(a, b, alternative=alternative)
    mean_diff = float(a.mean() - b.mean())
    better = "a" if mean_diff > 0 else "b" if mean_diff < 0 else "tie"
    return PairedResult(float(statistic), float(p_value), better)


def cross_environment(
    make_model: Callable[[], Any],
    environments: Mapping[str, tuple[Any, Any]],
    *,
    scoring: Callable[[Any, Any, Any], float],
) -> pl.DataFrame:
    """Train on each environment, score on every other — the does-it-generalize matrix.

    ``environments`` maps a name → ``(x, y)``; ``make_model()`` builds a fresh estimator per
    train environment; ``scoring(fitted, x, y)`` returns a metric (e.g.
    ``lambda m, x, y: evaluate.classification_metrics(y, m.predict(x), y_score=...)['roc_auc']``).
    Returns a long frame (``train``, ``test``, ``score``); the diagonal is in-domain performance,
    off-diagonal cells are transfer. A model that aces its own environment but collapses on the
    others is overfit to a regime (segment, period, geography) — read the row spread before
    trusting one number. The cross-segment / cross-period generalization check, generalized.
    """
    names = list(environments)
    rows = []
    for train_name in names:
        x_train, y_train = environments[train_name]
        fitted = make_model().fit(to_features(x_train), to_target(y_train))
        for test_name in names:
            x_test, y_test = environments[test_name]
            rows.append(
                {
                    "train": train_name,
                    "test": test_name,
                    "score": float(scoring(fitted, to_features(x_test), to_target(y_test))),
                }
            )
    return pl.DataFrame(rows)
