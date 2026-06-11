"""Convert Polars frames to the shapes sklearn expects (pandas features, 1-D target)."""

from __future__ import annotations

from typing import Any

import polars as pl


def to_features(x: Any) -> Any:
    """Feature matrix for sklearn — pandas, so name-based ColumnTransformer selection works."""
    return x.to_pandas() if isinstance(x, pl.DataFrame) else x


def to_target(y: Any) -> Any:
    """A 1-D target array for sklearn."""
    if isinstance(y, pl.Series):
        return y.to_numpy()
    if isinstance(y, pl.DataFrame):
        return y.to_numpy().ravel()
    return y
