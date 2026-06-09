"""Fit-on-train preprocessing pipelines (sklearn) — impute, scale, encode without leakage.

Build the preprocessor once, ``fit`` on the training split, then ``transform`` train and test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    *,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    scale: bool = True,
    impute: bool = True,
) -> ColumnTransformer:
    """ColumnTransformer: impute+scale numeric, impute+one-hot categorical (fit on train)."""
    numeric_steps: list[Any] = []
    categorical_steps: list[Any] = []
    if impute:
        numeric_steps.append(("impute", SimpleImputer(strategy="median")))
        categorical_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps.append(
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    )

    transformers: list[Any] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric)))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), list(categorical)))
    return ColumnTransformer(transformers, remainder="drop")
