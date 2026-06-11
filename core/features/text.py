"""Light text features — normalization, token counts, TF-IDF (no heavy NLP stack)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def normalize(df: pl.DataFrame, column: str, *, lower: bool = True) -> pl.DataFrame:
    """Strip, collapse whitespace, and optionally lowercase a string column."""
    expr = pl.col(column).str.strip_chars().str.replace_all(r"\s+", " ")
    if lower:
        expr = expr.str.to_lowercase()
    return df.with_columns(expr.alias(column))


def token_count(df: pl.DataFrame, column: str, *, output: str = "n_tokens") -> pl.DataFrame:
    """Add a whitespace-token count for a string column."""
    return df.with_columns(pl.col(column).str.split(" ").list.len().alias(output))


def tfidf(texts: Sequence[str], *, max_features: int = 500, **params: Any) -> tuple[Any, Any]:
    """Fit a TF-IDF vectorizer (sklearn); return (sparse matrix, fitted vectorizer)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, **params)
    return vectorizer.fit_transform(list(texts)), vectorizer
