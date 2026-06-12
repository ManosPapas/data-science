"""Causal inference — DiD, propensity matching & weighting, IV / non-compliance, uplift (ATE).

Matching and weighting remove *observed* confounding; DiD removes time-invariant unobserved
confounding; instruments handle confounded treatment when a valid instrument exists. Pick by which
assumption you can defend, not by which estimate you like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from core.analytics import stats


def difference_in_differences(
    control_before: float, control_after: float, treat_before: float, treat_after: float
) -> float:
    """DiD estimate: (treat_after - treat_before) - (control_after - control_before)."""
    return (treat_after - treat_before) - (control_after - control_before)


def uplift(treatment_outcome: ArrayLike, control_outcome: ArrayLike) -> float:
    """Average treatment effect: mean(treatment) - mean(control)."""
    treated = float(np.asarray(treatment_outcome, dtype=float).mean())
    control = float(np.asarray(control_outcome, dtype=float).mean())
    return treated - control


def propensity_scores(x: Any, treatment: ArrayLike) -> NDArray[np.float64]:
    """Estimate P(treatment | x) with logistic regression."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(x)
    model = LogisticRegression(max_iter=1000).fit(features, np.asarray(treatment))
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def match_on_propensity(
    scores: ArrayLike, treatment: ArrayLike, *, caliper: float = 0.05
) -> NDArray[np.int_]:
    """Nearest-neighbour match each treated unit to a control within ``caliper``.

    Returns, per treated unit (in index order), the matched control's row index, or -1 if none.
    """
    propensity = np.asarray(scores, dtype=float)
    flags = np.asarray(treatment).astype(int)
    treated = np.where(flags == 1)[0]
    controls = np.where(flags == 0)[0]
    matches = np.full(treated.size, -1, dtype=int)
    for position, treated_idx in enumerate(treated):
        if controls.size == 0:
            break
        distances = np.abs(propensity[controls] - propensity[treated_idx])
        nearest = int(np.argmin(distances))
        if distances[nearest] <= caliper:
            matches[position] = int(controls[nearest])
    return matches


def ipw_ate(outcome: ArrayLike, treatment: ArrayLike, propensity: ArrayLike) -> float:
    """Inverse-propensity-weighted ATE — weighting instead of matching for observed confounders.

    Each unit is weighted by 1/P(its own treatment | x), which rebalances the two groups to the
    same covariate mix (uses the normalized/Hájek form; propensities are clipped to [0.01, 0.99]
    so a near-0/1 score can't let one unit dominate). Keeps all rows, unlike matching — but is
    sensitive to a misspecified propensity model.
    """
    y = np.asarray(outcome, dtype=float)
    d = np.asarray(treatment).astype(float)
    p = np.clip(np.asarray(propensity, dtype=float), 0.01, 0.99)
    treated_weights = d / p
    control_weights = (1.0 - d) / (1.0 - p)
    treated_mean = float(np.sum(treated_weights * y) / np.sum(treated_weights))
    control_mean = float(np.sum(control_weights * y) / np.sum(control_weights))
    return treated_mean - control_mean


@dataclass(frozen=True)
class IttTotResult:
    """Experiment effects under non-compliance: assignment effect, uptake, complier effect."""

    itt: float
    compliance: float
    tot: float


def itt_tot(assigned: ArrayLike, treated: ArrayLike, outcome: ArrayLike) -> IttTotResult:
    """Intent-to-treat vs treatment-on-treated when not everyone assigned actually complies.

    ``itt`` = mean outcome difference by *assignment* (what shipping the policy really delivers);
    ``compliance`` = how much assignment moved actual treatment (the first stage); ``tot`` =
    itt / compliance — the Wald/IV estimate of the effect on compliers (LATE, the pure effect).
    Randomized assignment is the instrument, so this needs no confounder adjustment.
    """
    z = np.asarray(assigned).astype(float)
    d = np.asarray(treated).astype(float)
    y = np.asarray(outcome, dtype=float)
    itt = float(y[z == 1].mean() - y[z == 0].mean())
    compliance = float(d[z == 1].mean() - d[z == 0].mean())
    if abs(compliance) < 1e-12:
        raise ValueError("assignment did not move treatment uptake — TOT is unidentified")
    return IttTotResult(itt, compliance, itt / compliance)


def iv_effect(outcome: ArrayLike, treatment: ArrayLike, instrument: ArrayLike) -> float:
    """Instrumental-variable estimate of the treatment effect: cov(z, y) / cov(z, t).

    Use when treatment is self-selected/confounded but you have an instrument ``z`` that shifts
    treatment and affects the outcome *only* through it (exclusion restriction — argue it, you
    can't test it). Equivalent to two-stage least squares with a single instrument; raises when
    the instrument barely moves treatment (weak instruments make the ratio explode).
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=float)
    z = np.asarray(instrument, dtype=float)
    denominator = float(np.cov(z, t)[0, 1])
    if abs(denominator) < 1e-12:
        raise ValueError("weak instrument: cov(instrument, treatment) is ~0")
    return float(np.cov(z, y)[0, 1] / denominator)


def subgroup_effects(
    df: pl.DataFrame, *, outcome: str, treatment: str, segment: str
) -> pl.DataFrame:
    """Heterogeneous treatment effects: per-segment uplift with a Welch p-value.

    ``treatment`` is a 0/1 (or boolean) column. Answers "who benefits most?" — but slicing one
    experiment many ways multiplies false positives, so treat surprising subgroups as hypotheses
    to re-test, not results. Segments with fewer than 2 units in either arm are skipped.
    """
    schema: dict[str, Any] = {
        "segment": df.schema[segment],
        "n_control": pl.Int64,
        "n_treatment": pl.Int64,
        "control_mean": pl.Float64,
        "treatment_mean": pl.Float64,
        "effect": pl.Float64,
        "p_value": pl.Float64,
    }
    is_treated = pl.col(treatment).cast(pl.Boolean)
    rows: list[dict[str, Any]] = []
    for level in df.select(segment).unique().sort(segment).to_series().to_list():
        group = df.filter(pl.col(segment) == level)
        treated = group.filter(is_treated)[outcome].drop_nulls().to_numpy()
        control = group.filter(~is_treated)[outcome].drop_nulls().to_numpy()
        if treated.size < 2 or control.size < 2:
            continue
        rows.append(
            {
                "segment": level,
                "n_control": control.size,
                "n_treatment": treated.size,
                "control_mean": float(control.mean()),
                "treatment_mean": float(treated.mean()),
                "effect": float(treated.mean() - control.mean()),
                "p_value": stats.welch_t_test(control, treated).p_value,
            }
        )
    return pl.DataFrame(rows, schema=schema).sort("effect", descending=True)
