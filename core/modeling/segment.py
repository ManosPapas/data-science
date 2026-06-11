"""Segmentation — fit clusterers and reduce dimensions; the diagnostics live in ``viz.cluster``.

Heavy fitting happens here (compute); pass the arrays it returns to the charts (plot).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.modeling.frames import to_features


def make_clusterer(name: str = "kmeans", **params: Any) -> Any:
    """Build a clusterer: kmeans / minibatch_kmeans / dbscan / agglomerative / gaussian_mixture."""
    if name == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(**params)
    if name == "minibatch_kmeans":
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(**params)
    if name == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(**params)
    if name == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(**params)
    if name == "gaussian_mixture":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(**params)
    raise ValueError(f"unknown clusterer: {name}")


def elbow_scores(
    x: Any, k_values: Sequence[int], *, seed: int = 42
) -> tuple[list[int], list[float]]:
    """Fit KMeans for each k; return (k_values, inertias) for ``viz.cluster.elbow``."""
    from sklearn.cluster import KMeans

    features = np.asarray(to_features(x))
    ks = list(k_values)
    inertias: list[float] = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features)
        inertias.append(float(model.inertia_))
    return ks, inertias


def silhouette_scores(
    x: Any, k_values: Sequence[int], *, seed: int = 42
) -> tuple[list[int], list[float]]:
    """Fit KMeans for each k; return (k_values, mean silhouette) for ``viz.cluster.silhouette``."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = np.asarray(to_features(x))
    ks = list(k_values)
    scores: list[float] = []
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features)
        scores.append(float(silhouette_score(features, labels)))
    return ks, scores


def pca(x: Any, *, n_components: int = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit PCA; return (coords, explained_variance_ratio) for the ``viz.cluster`` charts."""
    from sklearn.decomposition import PCA

    features = np.asarray(to_features(x))
    model = PCA(n_components=n_components)
    coords = np.asarray(model.fit_transform(features), dtype=float)
    return coords, np.asarray(model.explained_variance_ratio_, dtype=float)
