"""Decision-support interpretation — counterfactuals, conformal intervals, confidence scores.

Three answers a model owes its stakeholders beyond a score: *what would change the outcome*
(counterfactual), *how far off might this prediction be* (conformal interval, distribution-free),
and *how sure is the classifier here* (confidence for triage/human-review routing). SHAP charts
(``viz.explain``) say which features mattered; the counterfactual says what to *do*.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from core.modeling.train import predict, predict_proba


def _score(model: Any, x: pl.DataFrame) -> NDArray[np.float64]:
    if hasattr(model, "predict_proba"):
        return np.asarray(predict_proba(model, x)[:, 1], dtype=float)
    return np.asarray(predict(model, x), dtype=float)


@dataclass(frozen=True)
class Counterfactual:
    """The smallest found change that flips the prediction past the target."""

    found: bool
    changes: dict[str, Any]
    score: float
    baseline_score: float
    candidates_evaluated: int


def counterfactual(
    model: Any,
    row: pl.DataFrame,
    *,
    candidates: Mapping[str, ArrayLike],
    target: float,
    direction: str = ">=",
    max_changes: int = 2,
    max_evaluations: int = 20_000,
) -> Counterfactual:
    """Search for the minimal feature change that pushes the score past ``target``.

    ``candidates`` maps each *actionable* feature to the values it may take — only include
    levers the business can actually pull (discount, plan, contact frequency), never immutable
    attributes; a counterfactual on tenure is an explanation, not an action. Greedy: all single
    changes first, then pairs (up to ``max_changes``=2), preferring fewer and smaller moves.
    Correlated features caveat: changing one feature while holding others fixed can describe an
    impossible customer — sanity-check the winning change against the data.
    """
    if direction not in (">=", "<="):
        raise ValueError("direction must be '>=' or '<='")
    if row.height != 1:
        raise ValueError("row must be a single-row frame")
    missing = [name for name in candidates if name not in row.columns]
    if missing:
        raise ValueError(f"candidate features not in the row: {missing}")

    baseline = float(_score(model, row)[0])

    def meets(score: float) -> bool:
        return score >= target if direction == ">=" else score <= target

    if meets(baseline):
        return Counterfactual(True, {}, baseline, baseline, 0)

    def normalized_move(name: str, value: Any) -> float:
        pool = np.asarray(candidates[name])
        current = row[name][0]
        if pool.dtype.kind in "if" and current is not None:
            span = float(pool.max() - pool.min()) or 1.0
            return abs(float(value) - float(current)) / span
        return 1.0  # categorical: any change costs one unit

    evaluated = 0
    for size in range(1, max_changes + 1):
        options: list[tuple[float, dict[str, Any]]] = []
        for names in combinations(candidates, size):
            pools = [np.asarray(candidates[name]).tolist() for name in names]
            for values in product(*pools):
                change = {n: v for n, v in zip(names, values, strict=True) if v != row[n][0]}
                if len(change) != size:
                    continue
                evaluated += 1
                if evaluated > max_evaluations:
                    raise ValueError(
                        "candidate grid too large — trim candidates or lower max_changes"
                    )
                variant = row.with_columns(
                    [pl.lit(v).cast(row.schema[n]).alias(n) for n, v in change.items()]
                )
                score = float(_score(model, variant)[0])
                if meets(score):
                    cost = sum(normalized_move(n, v) for n, v in change.items())
                    options.append((cost, {**change, "_score": score}))
        if options:
            cost, best = min(options, key=lambda pair: pair[0])
            score = float(best.pop("_score"))
            return Counterfactual(True, best, score, baseline, evaluated)
    return Counterfactual(False, {}, baseline, baseline, evaluated)


def conformal_intervals(
    model: Any,
    x_calibration: pl.DataFrame,
    y_calibration: ArrayLike,
    x_new: pl.DataFrame,
    *,
    alpha: float = 0.1,
) -> pl.DataFrame:
    """Distribution-free prediction intervals for any fitted regressor (split conformal).

    Guarantee: intervals contain the truth with probability ≥ 1-alpha on exchangeable data — no
    normality, no model trust required. ``x_calibration``/``y_calibration`` must be *held out
    from training* (use the validation split); the guarantee is marginal (on average), not
    per-row, and breaks under drift — re-calibrate on recent data periodically.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y_cal = np.asarray(y_calibration, dtype=float)
    n = y_cal.size
    minimum_rows = int(np.ceil(1.0 / alpha))
    if n < minimum_rows:
        raise ValueError(f"need at least {minimum_rows} calibration rows for this alpha")
    residuals = np.abs(np.asarray(predict(model, x_calibration), dtype=float) - y_cal)
    level = float(np.ceil((n + 1) * (1.0 - alpha))) / n
    margin = float(np.quantile(residuals, min(level, 1.0), method="higher"))
    point = np.asarray(predict(model, x_new), dtype=float)
    return pl.DataFrame({"prediction": point, "lower": point - margin, "upper": point + margin})


def confidence_score(probabilities: ArrayLike, *, method: str = "margin") -> NDArray[np.float64]:
    """Per-row classifier confidence in [0, 1] for triage: route low scores to a human.

    ``margin`` = top-1 minus top-2 probability (how clear the winner is); ``entropy`` = 1 -
    normalized entropy (how concentrated the whole distribution is). Confidence is only as
    honest as the probabilities — check ``viz.model.calibration`` before trusting the routing.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if method == "margin":
        ordered = np.sort(p, axis=1)
        return np.asarray(ordered[:, -1] - ordered[:, -2], dtype=float)
    if method == "entropy":
        clipped = np.clip(p, 1e-12, 1.0)
        entropy = -np.sum(clipped * np.log(clipped), axis=1)
        return np.asarray(1.0 - entropy / np.log(p.shape[1]), dtype=float)
    raise ValueError("method must be 'margin' or 'entropy'")
