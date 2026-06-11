"""Hyper-parameter search — grid and randomized (sklearn). Returns the fitted search object."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from core.modeling.frames import to_features, to_target


def grid_search(
    model: Any, param_grid: Any, x: Any, y: Any, *, cv: Any = 5, scoring: Any = None
) -> Any:
    """Exhaustive grid search; the result exposes ``.best_estimator_`` / ``.best_params_``."""
    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    search.fit(to_features(x), to_target(y))
    return search


def random_search(
    model: Any,
    param_distributions: Any,
    x: Any,
    y: Any,
    *,
    n_iter: int = 50,
    cv: Any = 5,
    scoring: Any = None,
    seed: int = 42,
) -> Any:
    """Randomized search over parameter distributions; returns the fitted RandomizedSearchCV."""
    search = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring, random_state=seed
    )
    search.fit(to_features(x), to_target(y))
    return search
