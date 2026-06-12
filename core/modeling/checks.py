"""Model behaviour validation — does the model respect business logic, not just the test set?

A model can score well and still be unshippable: demand that *rises* with price, risk that falls
as exposure grows, predictions that flip under measurement noise. These checks probe a fitted
model the way a domain reviewer would — sweep one feature, watch the prediction — and quantify
violations. Run them next to ``evaluate`` metrics before promoting any model; data-side rules
live in ``features.validate.check_rules``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from core.modeling.train import predict, predict_proba


def _score(model: Any, x: pl.DataFrame) -> NDArray[np.float64]:
    """Positive-class probability for classifiers, plain prediction otherwise."""
    if hasattr(model, "predict_proba"):
        return np.asarray(predict_proba(model, x)[:, 1], dtype=float)
    return np.asarray(predict(model, x), dtype=float)


def _sample(x: pl.DataFrame, n_rows: int, seed: int) -> pl.DataFrame:
    if x.height <= n_rows:
        return x
    return x.sample(n_rows, seed=seed)


@dataclass(frozen=True)
class MonotonicityResult:
    """Share of probed rows where the prediction moves against the expected direction."""

    feature: str
    direction: str
    violation_rate: float
    worst_gap: float
    rows_checked: int

    @property
    def passed(self) -> bool:
        return self.violation_rate == 0.0


def monotonicity(
    model: Any,
    x: pl.DataFrame,
    *,
    feature: str,
    direction: str = "increasing",
    n_grid: int = 11,
    n_rows: int = 200,
    tolerance: float = 1e-9,
    seed: int = 42,
) -> MonotonicityResult:
    """Sweep ``feature`` over its observed range per row; flag wrong-direction prediction moves.

    ``direction`` is what the *score* should do as the feature increases ("increasing" /
    "decreasing") — e.g. churn score increasing in support tickets, demand decreasing in price.
    Violations on a flexible model are usually real (interactions learned from confounded data);
    fix with monotonicity constraints (lightgbm/xgboost ``monotone_constraints``) or features,
    don't ship and hope.
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if feature not in x.columns:
        raise ValueError(f"feature {feature!r} not in the frame")
    sample = _sample(x, n_rows, seed)
    grid = np.unique(np.quantile(x[feature].drop_nulls().to_numpy(), np.linspace(0.0, 1.0, n_grid)))
    if grid.size < 2:
        raise ValueError(f"feature {feature!r} has no variation to sweep")
    dtype = x.schema[feature]
    swept = pl.concat(
        [sample.with_columns(pl.lit(float(v)).cast(dtype).alias(feature)) for v in grid]
    )
    scores = _score(model, swept).reshape(grid.size, sample.height)
    steps = np.diff(scores, axis=0)
    wrong = -steps if direction == "increasing" else steps
    violated = (wrong > tolerance).any(axis=0)
    return MonotonicityResult(
        feature=feature,
        direction=direction,
        violation_rate=float(violated.mean()),
        worst_gap=float(np.maximum(wrong, 0.0).max()),
        rows_checked=sample.height,
    )


def expected_directions(
    model: Any,
    x: pl.DataFrame,
    directions: Mapping[str, str],
    *,
    n_grid: int = 11,
    n_rows: int = 200,
    seed: int = 42,
) -> pl.DataFrame:
    """Run :func:`monotonicity` for every (feature → direction) rule; one row per rule.

    The shippable artifact: encode the business's sign expectations once, re-run on every
    retrain, and alert on regressions — model governance as a table.
    """
    rows = []
    for feature, direction in directions.items():
        result = monotonicity(
            model, x, feature=feature, direction=direction, n_grid=n_grid, n_rows=n_rows, seed=seed
        )
        rows.append(
            {
                "feature": feature,
                "direction": direction,
                "violation_rate": result.violation_rate,
                "worst_gap": result.worst_gap,
                "passed": result.passed,
            }
        )
    return pl.DataFrame(rows).sort("violation_rate", descending=True)


def prediction_bounds(
    model: Any,
    x: pl.DataFrame,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float]:
    """Share of predictions outside the plausible [lower, upper] business range.

    Negative prices, conversion above 100%, demand past physical capacity — out-of-range outputs
    mean extrapolation or target leakage. Returns min/max and the offending shares; clip at
    serving time only *after* understanding why.
    """
    scores = _score(model, x)
    return {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "share_below": float(np.mean(scores < lower)) if lower is not None else 0.0,
        "share_above": float(np.mean(scores > upper)) if upper is not None else 0.0,
    }


@dataclass(frozen=True)
class StabilityReport:
    """Prediction movement under small input noise — robustness of the scores."""

    mean_abs_change: float
    p95_abs_change: float
    prediction_range: float

    @property
    def relative_mean_change(self) -> float:
        """Mean change as a share of the prediction range — comparable across models."""
        return self.mean_abs_change / self.prediction_range if self.prediction_range else 0.0


def perturbation_stability(
    model: Any,
    x: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    scale: float = 0.05,
    n_repeats: int = 20,
    seed: int = 42,
) -> StabilityReport:
    """Jitter numeric features by ``scale``·std and measure how much predictions move.

    Inputs are measured with error; a model whose scores swing on noise-sized changes will
    thrash downstream decisions (offers granted on Monday, refused on Tuesday). Large
    ``p95_abs_change`` with healthy accuracy points at over-sharp decision boundaries —
    regularize, ensemble, or smooth before deploying.
    """
    numeric = [
        name
        for name, dtype in x.schema.items()
        if dtype.is_numeric() and (columns is None or name in columns)
    ]
    if not numeric:
        raise ValueError("no numeric columns to perturb")
    rng = np.random.default_rng(seed)
    baseline = _score(model, x)
    spread = {}
    for name in numeric:
        deviation = x[name].drop_nulls().std()
        spread[name] = float(deviation) if isinstance(deviation, int | float) else 0.0
    changes = []
    for _ in range(n_repeats):
        noisy = x.with_columns(
            [
                (pl.col(name) + rng.normal(0.0, scale * spread[name], x.height)).alias(name)
                for name in numeric
                if spread[name] > 0
            ]
        )
        changes.append(np.abs(_score(model, noisy) - baseline))
    deltas = np.concatenate(changes)
    return StabilityReport(
        mean_abs_change=float(deltas.mean()),
        p95_abs_change=float(np.percentile(deltas, 95)),
        prediction_range=float(baseline.max() - baseline.min()),
    )
