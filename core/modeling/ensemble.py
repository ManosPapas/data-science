"""Combine several models into one — voting and stacking, for both tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)


def make_voting(
    estimators: Sequence[tuple[str, Any]],
    *,
    task: str = "classification",
    weights: Sequence[float] | None = None,
    voting: str = "soft",
) -> Any:
    """Voting ensemble over named (name, estimator) pairs."""
    members = list(estimators)
    if task == "classification":
        return VotingClassifier(estimators=members, voting=voting, weights=weights)
    return VotingRegressor(estimators=members, weights=weights)


def make_stacking(
    estimators: Sequence[tuple[str, Any]],
    *,
    task: str = "classification",
    final_estimator: Any = None,
    cv: Any = 5,
) -> Any:
    """Stacking ensemble: base learners feed a meta-model (``final_estimator``)."""
    members = list(estimators)
    if task == "classification":
        return StackingClassifier(estimators=members, final_estimator=final_estimator, cv=cv)
    return StackingRegressor(estimators=members, final_estimator=final_estimator, cv=cv)
