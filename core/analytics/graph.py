"""Network analytics over edge lists — centrality, components, paths, trees, flows.

Commercial graphs are everywhere once you look: co-purchase networks (from ``analytics.basket``),
referral programs, supplier/logistics networks, money movement. Input everywhere is a Polars
edge-list frame (``source``, ``target``[, ``weight``]); compute is scipy.sparse.csgraph — no
extra dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _build(
    edges: pl.DataFrame,
    *,
    source: str,
    target: str,
    weight: str | None,
    directed: bool,
) -> tuple[Any, list[Any]]:
    """Adjacency matrix (scipy CSR) and the node list its indices refer to."""
    from scipy.sparse import coo_matrix

    if edges.is_empty():
        raise ValueError("edge list is empty")
    clean = edges.drop_nulls([source, target])
    nodes_series = pl.concat([clean[source], clean[target].rename(source)]).unique(
        maintain_order=True
    )
    nodes = nodes_series.to_list()
    index = {node: i for i, node in enumerate(nodes)}
    rows = np.array([index[v] for v in clean[source].to_list()])
    cols = np.array([index[v] for v in clean[target].to_list()])
    weights = (
        clean[weight].cast(pl.Float64).to_numpy() if weight is not None else np.ones(clean.height)
    )
    n = len(nodes)
    matrix = coo_matrix((weights, (rows, cols)), shape=(n, n))
    if not directed:
        matrix = matrix + matrix.T
    return matrix.tocsr(), nodes


def degree_centrality(
    edges: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    directed: bool = False,
) -> pl.DataFrame:
    """Connections per node: degree, weighted degree, and normalized centrality (degree/(n-1)).

    The first-pass importance read — hub products in a co-purchase graph, super-referrers in a
    referral network. Degree is local; for influence through the *structure* use :func:`pagerank`.
    """
    matrix, nodes = _build(edges, source=source, target=target, weight=weight, directed=directed)
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    out_degree = np.asarray(binary.sum(axis=1)).ravel()
    in_degree = np.asarray(binary.sum(axis=0)).ravel()
    out_weight = np.asarray(matrix.sum(axis=1)).ravel()
    in_weight = np.asarray(matrix.sum(axis=0)).ravel()
    n = len(nodes)
    if directed:
        frame = pl.DataFrame(
            {
                "node": pl.Series(nodes),
                "in_degree": in_degree,
                "out_degree": out_degree,
                "weighted_in": in_weight,
                "weighted_out": out_weight,
            }
        ).with_columns(
            ((pl.col("in_degree") + pl.col("out_degree")) / max(n - 1, 1)).alias("centrality")
        )
        return frame.sort("centrality", descending=True)
    # Symmetrized matrix counts each undirected edge in both triangles, so the row sum *is* the
    # plain degree.
    return (
        pl.DataFrame(
            {"node": pl.Series(nodes), "degree": out_degree, "weighted_degree": out_weight}
        )
        .with_columns((pl.col("degree") / max(n - 1, 1)).alias("centrality"))
        .sort("centrality", descending=True)
    )


def pagerank(
    edges: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    directed: bool = True,
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> pl.DataFrame:
    """Structural importance by random surfer (power iteration); scores sum to 1.

    A node ranks high when *important* nodes point at it — degree weighted by the quality of the
    neighbours. Damping is the restart probability; dangling nodes redistribute uniformly.
    """
    matrix, nodes = _build(edges, source=source, target=target, weight=weight, directed=directed)
    n = len(nodes)
    out_strength = np.asarray(matrix.sum(axis=1)).ravel()
    dangling = out_strength == 0
    inverse = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out_strength))
    transition = matrix.T.multiply(inverse).tocsr()  # column-stochastic on non-dangling nodes
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        spread = transition @ rank + rank[dangling].sum() / n
        updated = damping * spread + (1.0 - damping) / n
        if float(np.abs(updated - rank).sum()) < tol:
            rank = updated
            break
        rank = updated
    return pl.DataFrame({"node": pl.Series(nodes), "pagerank": rank}).sort(
        "pagerank", descending=True
    )


def connected_components(
    edges: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    directed: bool = False,
) -> pl.DataFrame:
    """Label each node's component (directed graphs use weak connectivity) with its size.

    Segmentation by reachability: distinct customer communities, disconnected product islands,
    fraud rings. Components of size 1-2 in a co-purchase graph are cross-sell dead ends.
    """
    from scipy.sparse.csgraph import connected_components as cc

    matrix, nodes = _build(edges, source=source, target=target, weight=None, directed=directed)
    _, labels = cc(matrix, directed=directed, connection="weak")
    frame = pl.DataFrame({"node": pl.Series(nodes), "component": labels})
    sizes = frame.group_by("component").len().rename({"len": "component_size"})
    return frame.join(sizes, on="component").sort(
        ["component_size", "component"], descending=[True, False]
    )


def shortest_paths(
    edges: pl.DataFrame,
    *,
    origin: Any,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    directed: bool = False,
) -> pl.DataFrame:
    """Dijkstra distances from ``origin`` to every node (unreachable → null).

    Weights are *costs/distances* here (lower = closer) — the opposite reading from similarity
    weights; invert similarities before routing on them.
    """
    from scipy.sparse.csgraph import dijkstra

    matrix, nodes = _build(edges, source=source, target=target, weight=weight, directed=directed)
    if origin not in nodes:
        raise ValueError(f"origin {origin!r} not present in the edge list")
    distances = dijkstra(matrix, directed=directed, indices=nodes.index(origin))
    return (
        pl.DataFrame({"node": pl.Series(nodes), "distance": distances})
        .with_columns(
            pl.when(pl.col("distance").is_infinite())
            .then(None)
            .otherwise(pl.col("distance"))
            .alias("distance")
        )
        .sort("distance", nulls_last=True)
    )


def minimum_spanning_tree_edges(
    edges: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
) -> pl.DataFrame:
    """The cheapest edge set connecting every node (undirected) — network design's lower bound.

    The minimal backbone for "connect all sites/markets at least cost"; anything beyond these
    edges buys redundancy, not reach.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    matrix, nodes = _build(edges, source=source, target=target, weight=weight, directed=False)
    tree = minimum_spanning_tree(matrix).tocoo()
    return pl.DataFrame(
        {
            "source": pl.Series([nodes[i] for i in tree.row]),
            "target": pl.Series([nodes[j] for j in tree.col]),
            "weight": tree.data,
        }
    ).sort("weight")


def max_flow(
    edges: pl.DataFrame,
    *,
    origin: Any,
    sink: Any,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    scale: float = 1.0,
) -> tuple[float, pl.DataFrame]:
    """Maximum flow from ``origin`` to ``sink`` through edge capacities; returns (value, flows).

    The throughput ceiling of a logistics/served-capacity network — the binding bottleneck is
    whichever cut saturates. Capacities must be integral; fractional capacities are scaled by
    ``scale``, rounded, and scaled back (pick ``scale`` to keep the rounding error immaterial).
    """
    from scipy.sparse.csgraph import maximum_flow

    matrix, nodes = _build(edges, source=source, target=target, weight=weight, directed=True)
    for label in (origin, sink):
        if label not in nodes:
            raise ValueError(f"node {label!r} not present in the edge list")
    capacities = matrix.copy()
    capacities.data = np.round(capacities.data * scale)
    capacities = capacities.astype(np.int64)
    result = maximum_flow(capacities, nodes.index(origin), nodes.index(sink))
    flow = result.flow.tocoo()
    keep = flow.data > 0
    flows = pl.DataFrame(
        {
            "source": pl.Series([nodes[i] for i in flow.row[keep]]),
            "target": pl.Series([nodes[j] for j in flow.col[keep]]),
            "flow": flow.data[keep] / scale,
        }
    ).sort("flow", descending=True)
    return float(result.flow_value / scale), flows
