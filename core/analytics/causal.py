"""Causal inference — DiD, matching/weighting, IV, RDD, synthetic control, uplift modeling.

Each design buys identification with a different assumption: matching/weighting remove *observed*
confounding, DiD removes time-invariant unobserved confounding, instruments handle confounded
treatment, RDD exploits a threshold rule, synthetic control builds an explicit counterfactual for
one treated unit. Pick by which assumption you can defend, not by which estimate you like.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

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


@dataclass(frozen=True)
class RddResult:
    """Sharp regression-discontinuity estimate at the cutoff."""

    effect: float
    std_err: float
    p_value: float
    n_left: int
    n_right: int


def regression_discontinuity(
    running: ArrayLike, outcome: ArrayLike, *, cutoff: float, bandwidth: float | None = None
) -> RddResult:
    """Sharp RDD: the outcome jump at a threshold rule (credit score, spend tier, exam mark).

    Units just below and just above the cutoff are as-good-as-randomized, so the fitted jump at
    the cutoff is causal — *locally*, for units near the threshold. Fits a linear trend with
    separate slopes per side within ``bandwidth`` of the cutoff (None = all data); shrink the
    bandwidth to trade bias for noise and check the estimate is stable across choices. Invalid if
    units can precisely manipulate their running variable (look for bunching at the cutoff).
    """
    import statsmodels.api as sm

    centred = np.asarray(running, dtype=float) - cutoff
    values = np.asarray(outcome, dtype=float)
    if bandwidth is not None:
        keep = np.abs(centred) <= bandwidth
        centred, values = centred[keep], values[keep]
    above = (centred >= 0).astype(float)
    n_right = int(above.sum())
    n_left = int(above.size - n_right)
    if min(n_left, n_right) < 3:
        raise ValueError("too few observations on one side of the cutoff")
    exog = np.column_stack([np.ones(centred.size), above, centred, above * centred])
    result = sm.OLS(values, exog).fit()
    return RddResult(
        float(result.params[1]), float(result.bse[1]), float(result.pvalues[1]), n_left, n_right
    )


@dataclass(frozen=True)
class SyntheticControlResult:
    """Donor weights, the synthetic counterfactual, and the estimated effect."""

    weights: pl.DataFrame
    effect: float
    pre_rmse: float
    synthetic_pre: NDArray[np.float64]
    synthetic_post: NDArray[np.float64]


def synthetic_control(
    treated_pre: ArrayLike,
    donors_pre: ArrayLike,
    treated_post: ArrayLike,
    donors_post: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
) -> SyntheticControlResult:
    """One treated unit (market, region, store) and no natural control: build one from donors.

    Finds non-negative, sum-to-one donor weights that best track the treated unit *before* the
    intervention; the post-period gap to that synthetic twin is the effect — an explicit
    counterfactual for "what would have happened otherwise". Donor matrices are (time, donors).
    Trust requires a small ``pre_rmse`` and donors untouched by the intervention; gauge
    significance by rerunning on each donor as a placebo.
    """
    from scipy.optimize import minimize

    pre_treated = np.asarray(treated_pre, dtype=float)
    pre_donors = np.asarray(donors_pre, dtype=float)
    post_treated = np.asarray(treated_post, dtype=float)
    post_donors = np.asarray(donors_post, dtype=float)
    n_donors = pre_donors.shape[1]

    def gap(weights: NDArray[np.float64]) -> float:
        return float(np.sum((pre_treated - pre_donors @ weights) ** 2))

    solution = minimize(
        gap,
        np.full(n_donors, 1.0 / n_donors),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    weights = np.asarray(solution.x, dtype=float)
    synthetic_pre = pre_donors @ weights
    synthetic_post = post_donors @ weights
    names = list(labels) if labels is not None else [f"donor_{i}" for i in range(n_donors)]
    return SyntheticControlResult(
        pl.DataFrame({"donor": names, "weight": weights}).sort("weight", descending=True),
        float(np.mean(post_treated - synthetic_post)),
        float(np.sqrt(np.mean((pre_treated - synthetic_pre) ** 2))),
        synthetic_pre,
        synthetic_post,
    )


class TLearner:
    """Uplift model (T-learner): predicts *who responds to treatment*, not who converts.

    Fits one outcome model on treated rows and one on control rows; predicted uplift is the
    difference. Rank customers by it to find persuadables (positive uplift) versus sure things
    and lost causes (~0) versus do-not-disturb (negative) — targeting on conversion probability
    instead wastes spend on people who'd convert anyway. Train on randomized data (or weight/
    match first); evaluate the *ranking* with :func:`qini_auc`, never with accuracy.
    """

    def __init__(self, model: Any) -> None:
        from sklearn.base import clone

        self.treated_model = clone(model)
        self.control_model = clone(model)

    def fit(self, x: Any, treatment: ArrayLike, outcome: ArrayLike) -> Self:
        features = np.asarray(x, dtype=float)
        treated = np.asarray(treatment).astype(bool)
        y = np.asarray(outcome)
        self.treated_model.fit(features[treated], y[treated])
        self.control_model.fit(features[~treated], y[~treated])
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        features = np.asarray(x, dtype=float)
        treated = self._expected(self.treated_model, features)
        control = self._expected(self.control_model, features)
        return np.asarray(treated - control, dtype=float)

    @staticmethod
    def _expected(model: Any, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        return np.asarray(model.predict(features), dtype=float)


def qini_points(
    outcome: ArrayLike, treatment: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Qini curve: incremental successes vs share of the population targeted, ranked by uplift.

    At each depth the curve shows treated successes minus control successes (scaled to the
    treated count) among the top-scored units — what targeting that slice *adds* over not
    treating. A useful model bows above the straight random-targeting line from (0, 0) to the
    full-population endpoint; plot with a line chart plus that diagonal.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment).astype(float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    y_sorted, t_sorted = y[order], t[order]
    cum_treated = np.cumsum(t_sorted)
    cum_control = np.cumsum(1.0 - t_sorted)
    treated_successes = np.cumsum(y_sorted * t_sorted)
    control_successes = np.cumsum(y_sorted * (1.0 - t_sorted))
    ratio = np.divide(
        cum_treated, cum_control, out=np.zeros_like(cum_treated), where=cum_control > 0
    )
    fractions = np.arange(1, y.size + 1) / y.size
    return fractions, treated_successes - control_successes * ratio


def qini_auc(outcome: ArrayLike, treatment: ArrayLike, scores: ArrayLike) -> float:
    """Area between the Qini curve and random targeting — bigger = better uplift *ranking*.

    The uplift world's AUC: ~0 means the model ranks no better than chance, negative means worse.
    Compare uplift models on this — a great outcome classifier can still be a useless uplift
    ranker, because predicting conversion is not predicting persuasion.
    """
    fractions, qini = qini_points(outcome, treatment, scores)
    random_line = qini[-1] * fractions
    return float(np.trapezoid(qini - random_line, fractions))


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
