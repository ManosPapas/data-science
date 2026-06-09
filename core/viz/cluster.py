"""Clustering & dimensionality-reduction diagnostics.

Charts take precomputed arrays — fit PCA/KMeans/linkage in analytics/modeling and pass the results
here (compute is not the chart's job).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from core.viz.base import chart


@chart(title="Cumulative explained variance")
def explained_variance(ax: Axes, explained_variance_ratio: ArrayLike) -> None:
    """Individual (bars) and cumulative (step) explained variance from a fitted PCA."""
    ratio = np.asarray(explained_variance_ratio, dtype=float)
    components = np.arange(1, ratio.size + 1)
    ax.bar(components, ratio, alpha=0.5, label="individual")
    ax.step(components, np.cumsum(ratio), where="mid", label="cumulative")
    ax.set(xlabel="principal component", ylabel="explained variance ratio")
    ax.legend(loc="center right")


@chart(title="Elbow curve")
def elbow(ax: Axes, k_values: ArrayLike, inertias: ArrayLike) -> None:
    """Inertia (within-cluster sum of squares) against the number of clusters."""
    ax.plot(np.asarray(k_values), np.asarray(inertias, dtype=float), marker="o")
    ax.set(xlabel="number of clusters (k)", ylabel="inertia")


@chart(title="Silhouette by k")
def silhouette(ax: Axes, k_values: ArrayLike, scores: ArrayLike) -> None:
    """Mean silhouette score against the number of clusters."""
    ax.plot(np.asarray(k_values), np.asarray(scores, dtype=float), marker="o", color="tab:green")
    ax.set(xlabel="number of clusters (k)", ylabel="mean silhouette")


@chart(title="Silhouette plot")
def silhouette_plot(ax: Axes, labels: ArrayLike, silhouette_values: ArrayLike) -> None:
    """Per-sample silhouette 'knife' plot from precomputed ``silhouette_samples`` values."""
    label_array = np.asarray(labels)
    values = np.asarray(silhouette_values, dtype=float)
    y_lower = 0
    for cluster in np.unique(label_array):
        cluster_values = np.sort(values[label_array == cluster])
        size = cluster_values.size
        ax.fill_betweenx(np.arange(y_lower, y_lower + size), 0, cluster_values, alpha=0.7)
        ax.text(-0.02, y_lower + size / 2, str(cluster))
        y_lower += size
    ax.axvline(float(values.mean()), linestyle="--", color="red")
    ax.set(xlabel="silhouette coefficient", ylabel="samples grouped by cluster")


@chart(title="Clusters")
def cluster_scatter(ax: Axes, coords: ArrayLike, labels: ArrayLike) -> None:
    """2-D scatter of points (e.g. PCA/UMAP coords) coloured by cluster label."""
    points = np.asarray(coords, dtype=float)
    sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=np.asarray(labels), palette="tab10", ax=ax)
    ax.set(xlabel="dim 1", ylabel="dim 2")


@chart(title="Dendrogram")
def dendrogram(ax: Axes, linkage_matrix: ArrayLike, *, labels: Sequence[str] | None = None) -> None:
    """Hierarchical-clustering tree from a precomputed scipy linkage matrix."""
    scipy_dendrogram(
        np.asarray(linkage_matrix, dtype=float),
        ax=ax,
        labels=list(labels) if labels is not None else None,
    )
    ax.set(xlabel="sample", ylabel="distance")
