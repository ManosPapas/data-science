"""Recommendation — item-item collaborative filtering on an interactions frame.

"Customers who bought X also bought…" from nothing but (user, item[, rating]) tuples: items are
similar when the same users engage with both (cosine over the user dimension). Item-item is the
production classic — item neighbourhoods are stabler than user tastes and recommendations come
with a "because you bought X" explanation for free. Evaluate rankings with
``modeling.evaluate.ranking_metrics``; popularity is the baseline to beat.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import polars as pl


class ItemItemRecommender:
    """Cosine item-item recommender. ``fit`` on interactions, then ``recommend``/``similar_items``.

    Implicit feedback (no ``rating`` column) scores every interaction 1. Cold items/users —
    unseen at fit time — get no scores: handle them with a popularity fallback upstream.
    """

    def __init__(self) -> None:
        self._users: list[Any] = []
        self._items: list[Any] = []
        self._user_index: dict[Any, int] = {}
        self._item_index: dict[Any, int] = {}
        self._interactions: Any = None  # users x items CSR
        self._similarity: Any = None  # items x items CSR, zero diagonal

    def fit(self, df: pl.DataFrame, *, user: str, item: str, rating: str | None = None) -> Self:
        """Build the user-by-item matrix and the item-item cosine similarity."""
        from scipy.sparse import coo_matrix

        clean = df.drop_nulls([user, item])
        if clean.is_empty():
            raise ValueError("no interactions to fit on")
        self._users = clean[user].unique(maintain_order=True).to_list()
        self._items = clean[item].unique(maintain_order=True).to_list()
        self._user_index = {u: i for i, u in enumerate(self._users)}
        self._item_index = {p: i for i, p in enumerate(self._items)}
        rows = np.array([self._user_index[u] for u in clean[user].to_list()])
        cols = np.array([self._item_index[p] for p in clean[item].to_list()])
        values = clean[rating].cast(pl.Float64).to_numpy() if rating else np.ones(clean.height)
        matrix = coo_matrix(
            (values, (rows, cols)), shape=(len(self._users), len(self._items))
        ).tocsr()

        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        normalized = matrix.multiply(1.0 / norms).tocsc()
        similarity = (normalized.T @ normalized).tolil()
        similarity.setdiag(0.0)
        self._interactions = matrix
        self._similarity = similarity.tocsr()
        return self

    def _check_fitted(self) -> None:
        if self._similarity is None:
            raise ValueError("call fit() first")

    def similar_items(self, item: Any, *, k: int = 10) -> pl.DataFrame:
        """Top-k most similar items — the 'frequently bought together' shelf for one product."""
        self._check_fitted()
        if item not in self._item_index:
            raise ValueError(f"unknown item {item!r}")
        scores = self._similarity[self._item_index[item]].toarray().ravel()
        order = np.argsort(scores)[::-1][:k]
        order = order[scores[order] > 0]
        return pl.DataFrame(
            {
                "item": pl.Series([self._items[i] for i in order]),
                "similarity": scores[order],
            }
        )

    def recommend(self, user: Any, *, k: int = 10, exclude_seen: bool = True) -> pl.DataFrame:
        """Top-k items for a user: similarity-weighted sum over what they already engaged with."""
        self._check_fitted()
        if user not in self._user_index:
            raise ValueError(f"unknown user {user!r}")
        row = self._interactions[self._user_index[user]]
        scores = np.asarray((row @ self._similarity).todense()).ravel()
        if exclude_seen:
            scores[row.indices] = -np.inf
        order = np.argsort(scores)[::-1][:k]
        order = order[np.isfinite(scores[order]) & (scores[order] > 0)]
        return pl.DataFrame(
            {"item": pl.Series([self._items[i] for i in order]), "score": scores[order]}
        )


def popularity_baseline(df: pl.DataFrame, *, item: str, k: int = 10) -> pl.DataFrame:
    """Most-interacted items — the baseline every recommender must beat to earn its complexity."""
    return (
        df.group_by(item)
        .len()
        .rename({"len": "interactions", item: "item"})
        .sort("interactions", descending=True)
        .head(k)
    )
