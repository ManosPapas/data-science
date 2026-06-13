"""Fit-on-train preprocessing pipelines (sklearn) — impute, scale, encode without leakage.

Build the preprocessor once, ``fit`` on the training split, then ``transform`` train and test.
Each concern is a swappable strategy: imputer ('mean'/'median'/'knn'/'iterative'), scaler
('standard'/'minmax'/'robust'), encoder ('onehot'/'ordinal'/'target').
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers the imputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


def make_imputer(strategy: str = "median", *, seed: int = 42) -> Any:
    """Numeric imputer: 'mean' / 'median' (simple), 'knn' (neighbour mean), 'iterative' (MICE).

    Simple statistics suit values missing completely at random (MCAR). When missingness relates to
    other columns (MAR — check ``stats.missingness_dependence``), 'knn' and 'iterative' impute
    conditionally on them; 'iterative' regresses each column on the rest in rounds. If missingness
    itself carries signal, also keep a flag (``clean.add_missing_indicators``).
    """
    if strategy in {"mean", "median", "most_frequent"}:
        return SimpleImputer(strategy=strategy)
    if strategy == "knn":
        return KNNImputer()
    if strategy == "iterative":
        return IterativeImputer(random_state=seed)
    raise ValueError(f"unknown imputer strategy: {strategy}")


def make_scaler(strategy: str = "standard") -> Any:
    """Scaler: 'standard' (z-score), 'minmax' (0-1 normalization), 'robust' (median/IQR).

    Standardize for linear models, SVMs, and PCA (they assume centred, comparable scales);
    normalize to 0-1 for distance-based models and neural nets; go robust when outliers would
    otherwise dominate the scale. Trees don't care either way.
    """
    if strategy == "standard":
        return StandardScaler()
    if strategy == "minmax":
        return MinMaxScaler()
    if strategy == "robust":
        return RobustScaler()
    raise ValueError(f"unknown scaler strategy: {strategy}")


def make_power_transformer(method: str = "yeo-johnson", *, standardize: bool = True) -> Any:
    """Variance-stabilizing power transform that *learns* its exponent on the training data.

    For skewed positive amounts (revenue, durations, counts) where a fixed ``transform.log1p`` is
    too blunt: Box-Cox / Yeo-Johnson fit the optimal power to make the column as normal as
    possible, which steadies variance and straightens relationships for linear models, SVMs, and
    distance methods. ``'yeo-johnson'`` handles zeros and negatives (the safe default);
    ``'box-cox'`` needs strictly positive input but is the classic. Being fit, it belongs here
    (train-only, applied to the rest) — unlike the stateless ``transform.log1p``. Trees don't need
    it. Decreasing order of reach for skew: log1p (quick) → power transform (fitted) → model the
    distribution directly (``stats.fit_distribution``).
    """
    if method not in {"yeo-johnson", "box-cox"}:
        raise ValueError("method must be 'yeo-johnson' or 'box-cox'")
    return PowerTransformer(method=method, standardize=standardize)


def make_encoder(strategy: str = "onehot") -> Any:
    """Categorical encoder: 'onehot' (nominal), 'ordinal' (integer codes), 'target' (mean target).

    One-hot for unordered categories (no artificial order, but explodes with cardinality).
    Ordinal codes are compact but impose an order — sklearn picks it alphabetically, so for real
    ordinal scales build an ``OrdinalEncoder(categories=[...])`` yourself. Target encoding handles
    high cardinality in one column; sklearn cross-fits it internally to limit target leakage (it
    needs ``y`` at fit time). Group rare levels first with ``transform.group_rare``.
    """
    if strategy == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if strategy == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if strategy == "target":
        return TargetEncoder()
    raise ValueError(f"unknown encoder strategy: {strategy}")


def make_preprocessor(
    *,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    scale: bool = True,
    impute: bool = True,
    imputer: str = "median",
    scaler: str = "standard",
    encoder: str = "onehot",
) -> ColumnTransformer:
    """ColumnTransformer: impute+scale numeric, impute+encode categorical (fit on train only)."""
    numeric_steps: list[Any] = []
    categorical_steps: list[Any] = []
    if impute:
        numeric_steps.append(("impute", make_imputer(imputer)))
        categorical_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    if scale:
        numeric_steps.append(("scale", make_scaler(scaler)))
    categorical_steps.append(("encode", make_encoder(encoder)))

    transformers: list[Any] = []
    if numeric:
        pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"
        transformers.append(("numeric", pipeline, list(numeric)))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), list(categorical)))
    return ColumnTransformer(transformers, remainder="drop")
