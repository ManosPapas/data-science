"""Hierarchical forecast reconciliation — make total/region/product forecasts add up.

Forecast every node of a hierarchy independently and the levels will disagree: the sum of the
regions won't equal the company line. Reconciliation restores coherence — and the projection
methods (``ols``) usually *improve* accuracy by pooling information across levels, rather than
just tidying the arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def _descendant_leaves(
    node: str, hierarchy: Mapping[str, Sequence[str]], leaves: list[str], depth: int = 0
) -> list[str]:
    if depth > 64:
        raise ValueError("hierarchy is too deep or cyclic")
    if node not in hierarchy:
        return [node]
    found: list[str] = []
    for child in hierarchy[node]:
        found.extend(_descendant_leaves(child, hierarchy, leaves, depth + 1))
    return found


def summing_matrix(
    hierarchy: Mapping[str, Sequence[str]],
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    """The S matrix mapping leaf values to every node; returns (S, nodes, leaves).

    ``hierarchy`` maps parent → children, e.g. ``{"total": ["EU", "NA"], "EU": ["UK", "DE"]}``.
    Node order is parents in insertion order followed by the leaves.
    """
    parents = list(hierarchy)
    children = {child for kids in hierarchy.values() for child in kids}
    leaves = [c for kids in hierarchy.values() for c in kids if c not in hierarchy]
    # preserve first-seen order, dedupe
    leaves = list(dict.fromkeys(leaves))
    roots = [p for p in parents if p not in children]
    if not roots:
        raise ValueError("hierarchy has no root — it contains a cycle")
    nodes = parents + leaves
    matrix = np.zeros((len(nodes), len(leaves)))
    leaf_index = {leaf: i for i, leaf in enumerate(leaves)}
    for row, node in enumerate(nodes):
        for leaf in _descendant_leaves(node, hierarchy, leaves):
            matrix[row, leaf_index[leaf]] = 1.0
    return matrix, nodes, leaves


def coherence_error(
    forecasts: Mapping[str, ArrayLike], hierarchy: Mapping[str, Sequence[str]]
) -> pl.DataFrame:
    """How badly each parent disagrees with the sum of its children (mean |gap| per node).

    The pre-reconciliation diagnostic: large gaps at one node mean its forecast (or its
    children's) deserves a look before any projection hides the disagreement.
    """
    rows = []
    for parent, kids in hierarchy.items():
        parent_values = np.asarray(forecasts[parent], dtype=float)
        child_sum = np.sum([np.asarray(forecasts[c], dtype=float) for c in kids], axis=0)
        rows.append(
            {
                "node": parent,
                "mean_abs_gap": float(np.mean(np.abs(parent_values - child_sum))),
                "mean_gap": float(np.mean(parent_values - child_sum)),
            }
        )
    return pl.DataFrame(rows).sort("mean_abs_gap", descending=True)


def reconcile(
    forecasts: Mapping[str, ArrayLike],
    hierarchy: Mapping[str, Sequence[str]],
    *,
    method: str = "ols",
    proportions: Mapping[str, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Coherent forecasts for every node from (possibly incoherent) base forecasts.

    - ``ols``: project all base forecasts onto the coherent subspace (MinT with identity
      weights) — every level's information is used; the default.
    - ``bottom_up``: trust the leaves, sum upward — safe when leaf forecasts are strong and
      aggregates are afterthoughts; noisy when leaves are sparse.
    - ``top_down``: trust the total, split by ``proportions`` over leaves (defaults to the
      shares implied by the leaf base forecasts) — stable aggregate, but per-leaf accuracy is
      only as good as the split.
    """
    s, nodes, leaves = summing_matrix(hierarchy)
    horizon = {np.asarray(v, dtype=float).size for v in forecasts.values()}
    if len(horizon) != 1:
        raise ValueError("all forecasts must share the same horizon")

    if method == "bottom_up":
        missing = [leaf for leaf in leaves if leaf not in forecasts]
        if missing:
            raise ValueError(f"bottom_up needs every leaf forecast; missing {missing}")
        bottom = np.vstack([np.asarray(forecasts[leaf], dtype=float) for leaf in leaves])
    elif method == "top_down":
        roots = [n for n in hierarchy if n not in {c for k in hierarchy.values() for c in k}]
        if len(roots) != 1:
            raise ValueError("top_down needs exactly one root node")
        total = np.asarray(forecasts[roots[0]], dtype=float)
        if proportions is None:
            base = np.array(
                [float(np.asarray(forecasts[leaf], dtype=float).mean()) for leaf in leaves]
            )
            if base.sum() <= 0:
                raise ValueError("cannot derive proportions from non-positive leaf forecasts")
            shares = base / base.sum()
        else:
            shares = np.array([float(proportions[leaf]) for leaf in leaves])
            if not np.isclose(shares.sum(), 1.0):
                raise ValueError("proportions over leaves must sum to 1")
        bottom = shares[:, None] * total[None, :]
    elif method == "ols":
        missing = [node for node in nodes if node not in forecasts]
        if missing:
            raise ValueError(f"ols reconciliation needs every node forecast; missing {missing}")
        stacked = np.vstack([np.asarray(forecasts[node], dtype=float) for node in nodes])
        bottom, _, _, _ = np.linalg.lstsq(s, stacked, rcond=None)
    else:
        raise ValueError("method must be 'ols', 'bottom_up', or 'top_down'")

    coherent = s @ bottom
    return {node: np.asarray(coherent[i], dtype=float) for i, node in enumerate(nodes)}
