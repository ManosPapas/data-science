"""Distance & similarity metrics — how far apart are two points, or all pairs.

The measure under every neighbourhood method (KNN, clustering, anomaly, matching, recommenders).
Picking the right one is a modelling decision, not a detail: Euclidean assumes comparable scales
(standardize first), cosine ignores magnitude (text/behaviour vectors), Mahalanobis accounts for
correlated features (the statistically honest multivariate distance), Jaccard/Hamming are for
sets/binary codes. Thin, typed wrappers over scipy/sklearn so the call site names the choice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_VECTOR_METRICS = {"euclidean", "cityblock", "cosine", "minkowski", "jaccard", "hamming"}
_ALIASES = {"manhattan": "cityblock", "l1": "cityblock", "l2": "euclidean"}


def vector_distance(
    a: ArrayLike, b: ArrayLike, *, metric: str = "euclidean", p: float = 2.0
) -> float:
    """Distance between two 1-D vectors. ``metric`` is euclidean/manhattan/cosine/minkowski/...

    euclidean = straight-line (scale-sensitive — standardize first); manhattan = grid/L1 (robust
    to outliers in any single dimension); cosine = angle only, ignores magnitude (the text /
    behaviour-vector default); minkowski with ``p`` interpolates (p=1 manhattan, p=2 euclidean);
    jaccard / hamming for binary or set-membership vectors.
    """
    from scipy.spatial import distance

    metric = _ALIASES.get(metric, metric)
    if metric not in _VECTOR_METRICS:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(_VECTOR_METRICS)}")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    func = getattr(distance, metric)
    return float(func(x, y, p) if metric == "minkowski" else func(x, y))


def pairwise(x: ArrayLike, *, metric: str = "euclidean", p: float = 2.0) -> NDArray[np.float64]:
    """All-pairs distance matrix for the rows of ``x`` — the input to clustering / kNN / MDS.

    Same ``metric`` choices as :func:`vector_distance`. Standardize features first for euclidean /
    minkowski so no single large-scale column dominates the geometry.
    """
    from scipy.spatial.distance import pdist, squareform

    resolved = _ALIASES.get(metric, metric)
    data = np.asarray(x, dtype=float)
    kwargs = {"p": p} if resolved == "minkowski" else {}
    return np.asarray(squareform(pdist(data, metric=resolved, **kwargs)), dtype=float)


def mahalanobis(point: ArrayLike, data: ArrayLike, *, ridge: float = 1e-6) -> float:
    """Mahalanobis distance of ``point`` from the centre of ``data`` — scale- and correlation-aware.

    The statistically honest multivariate distance: it whitens by the inverse covariance, so
    correlated or differently-scaled features are weighted correctly (a point 3 sd out along a
    tight axis is "far" even if its raw Euclidean distance is small). The natural multivariate
    outlier / novelty score (square it for a χ²-distributed value under normality). ``ridge``
    stabilizes the covariance inverse when features are near-collinear.
    """
    from scipy.spatial.distance import mahalanobis as _maha

    matrix = np.asarray(data, dtype=float)
    mean = matrix.mean(axis=0)
    cov = np.cov(matrix, rowvar=False)
    cov = np.atleast_2d(cov) + ridge * np.eye(matrix.shape[1])
    return float(_maha(np.asarray(point, dtype=float), mean, np.linalg.inv(cov)))


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity in [-1, 1] (1 - cosine distance) — direction agreement, magnitude-blind.

    The complement of :func:`vector_distance`'s cosine metric, framed as similarity: 1 = same
    direction, 0 = orthogonal, -1 = opposite. The default for comparing text/behaviour vectors
    where how *much* matters less than *what mix*.
    """
    return 1.0 - vector_distance(a, b, metric="cosine")
