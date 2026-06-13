"""Network drawing — a force-directed layout for edge lists, no extra dependency.

The eyes for ``analytics.graph``: co-purchase maps, referral webs, flow networks. A
Fruchterman-Reingold spring layout pushes all nodes apart and pulls connected ones together;
node size scales with degree, edge width with weight. Legible up to a few hundred nodes —
filter the edge list (e.g. by lift or weight) before drawing, not after.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray

from core.viz.base import chart

MAX_NODES = 500


def _spring_layout(
    n: int,
    edge_index: NDArray[np.int_],
    *,
    iterations: int,
    seed: int,
) -> NDArray[np.float64]:
    """Fruchterman-Reingold positions in the unit square."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.5, 0.5, (n, 2))
    k = 1.0 / np.sqrt(n)  # ideal pairwise distance
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    for _ in range(iterations):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, 1.0)
        # repulsion between every pair; attraction along edges only
        force = (k**2 / distance**2)[:, :, None] * delta
        displacement = force.sum(axis=1)
        edge_delta = positions[edge_index[:, 0]] - positions[edge_index[:, 1]]
        edge_distance = np.linalg.norm(edge_delta, axis=1, keepdims=True)
        edge_distance[edge_distance == 0] = 1e-9
        pull = edge_delta * (edge_distance / k)
        np.add.at(displacement, edge_index[:, 0], -pull)
        np.add.at(displacement, edge_index[:, 1], pull)
        length = np.linalg.norm(displacement, axis=1, keepdims=True)
        length[length == 0] = 1e-9
        positions += displacement / length * np.minimum(length, temperature)
        temperature -= cooling
    return np.asarray(positions, dtype=float)


@chart(title="Network")
def network(
    ax: Axes,
    edges: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    labels: bool = True,
    iterations: int = 120,
    seed: int = 42,
    max_nodes: int = MAX_NODES,
) -> None:
    """Draw an edge list as a spring-layout graph: hubs central, communities clustered.

    Node size encodes degree, edge width encodes ``weight``. The layout is qualitative —
    distances are suggestive, not measurements (read exact structure off ``analytics.graph``
    tables); fix ``seed`` for a reproducible picture. ``max_nodes`` guards legibility — raise it
    deliberately for a large-format export rather than relying on the default.
    """
    from matplotlib.collections import LineCollection

    clean = edges.drop_nulls([source, target])
    nodes = pl.concat([clean[source], clean[target].rename(source)]).unique(maintain_order=True)
    names = nodes.to_list()
    if len(names) > max_nodes:
        raise ValueError(
            f"{len(names)} nodes won't draw legibly — filter the edge list or raise max_nodes "
            f"(currently {max_nodes})"
        )
    index = {name: i for i, name in enumerate(names)}
    edge_index = np.array(
        [[index[a], index[b]] for a, b in zip(clean[source], clean[target], strict=True)]
    )
    positions = _spring_layout(len(names), edge_index, iterations=iterations, seed=seed)

    weights = (
        clean[weight].cast(pl.Float64).to_numpy() if weight is not None else np.ones(clean.height)
    )
    widths = 0.5 + 2.5 * (weights / weights.max())
    segments = np.stack([positions[edge_index[:, 0]], positions[edge_index[:, 1]]], axis=1)
    ax.add_collection(LineCollection(list(segments), colors="#bbbbbb", linewidths=list(widths)))

    degree = np.bincount(edge_index.ravel(), minlength=len(names))
    sizes = 80.0 + 600.0 * degree / max(int(degree.max()), 1)
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=sizes,
        color=sns.color_palette()[0],
        zorder=3,
        edgecolor="white",
        linewidth=1.0,
    )
    if labels:
        for name, (x, y) in zip(names, positions, strict=True):
            ax.annotate(
                str(name), (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9, zorder=4
            )
    ax.set_axis_off()
    ax.margins(0.15)
