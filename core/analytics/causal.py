"""Causal inference — difference-in-differences, propensity-score matching, uplift (ATE)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def difference_in_differences(
    control_before: float, control_after: float, treat_before: float, treat_after: float
) -> float:
    """DiD estimate: (treat_after - treat_before) - (control_after - control_before)."""
    return (treat_after - treat_before) - (control_after - control_before)


def uplift(treatment_outcome: ArrayLike, control_outcome: ArrayLike) -> float:
    """Average treatment effect: mean(treatment) - mean(control)."""
    treated = float(np.asarray(treatment_outcome, dtype=float).mean())
    control = float(np.asarray(control_outcome, dtype=float).mean())
    return treated - control


def propensity_scores(x: Any, treatment: ArrayLike) -> NDArray[np.float64]:
    """Estimate P(treatment | x) with logistic regression."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(x)
    model = LogisticRegression(max_iter=1000).fit(features, np.asarray(treatment))
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def match_on_propensity(
    scores: ArrayLike, treatment: ArrayLike, *, caliper: float = 0.05
) -> NDArray[np.int_]:
    """Nearest-neighbour match each treated unit to a control within ``caliper``.

    Returns, per treated unit (in index order), the matched control's row index, or -1 if none.
    """
    propensity = np.asarray(scores, dtype=float)
    flags = np.asarray(treatment).astype(int)
    treated = np.where(flags == 1)[0]
    controls = np.where(flags == 0)[0]
    matches = np.full(treated.size, -1, dtype=int)
    for position, treated_idx in enumerate(treated):
        if controls.size == 0:
            break
        distances = np.abs(propensity[controls] - propensity[treated_idx])
        nearest = int(np.argmin(distances))
        if distances[nearest] <= caliper:
            matches[position] = int(controls[nearest])
    return matches
