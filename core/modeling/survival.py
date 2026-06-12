"""Survival analysis — *when* churn/failure happens, not just whether (censoring done right).

Classification throws away timing and mishandles customers who simply haven't churned *yet*
(censored, not negative). Kaplan-Meier estimates the survival curve using every account; Cox
regression turns covariates into hazard ratios; restricted mean survival prices retention time
for CLV. Durations must share one origin definition (signup → churn/today).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


def _durations(
    durations: ArrayLike, events: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    if t.shape != e.shape:
        raise ValueError("durations and events must have the same length")
    if np.any(t < 0):
        raise ValueError("durations must be non-negative")
    if not np.isin(e, (0.0, 1.0)).all():
        raise ValueError("events must be 0 (censored) or 1 (event occurred)")
    if e.sum() == 0:
        raise ValueError("no events observed — the survival curve is unidentified")
    return t, e


def kaplan_meier(
    durations: ArrayLike, events: ArrayLike, *, confidence: float = 0.95
) -> pl.DataFrame:
    """Survival curve S(t) with pointwise CIs: P(still alive/subscribed past t).

    ``events``: 1 = churned/failed at ``duration``, 0 = censored (still active at last sight —
    they contribute exposure up to then, which is exactly what naive churn rates get wrong).
    Read retention at any tenure directly off the curve; compare segments by plotting both.
    """
    from scipy.stats import norm
    from statsmodels.duration.survfunc import SurvfuncRight

    t, e = _durations(durations, events)
    fit = SurvfuncRight(t, e)
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    survival = np.asarray(fit.surv_prob, dtype=float)
    std_err = np.asarray(fit.surv_prob_se, dtype=float)
    return pl.DataFrame(
        {
            "time": np.asarray(fit.surv_times, dtype=float),
            "at_risk": np.asarray(fit.n_risk, dtype=float),
            "survival": survival,
            "ci_low": np.clip(survival - z * std_err, 0.0, 1.0),
            "ci_high": np.clip(survival + z * std_err, 0.0, 1.0),
        }
    )


def survival_at(durations: ArrayLike, events: ArrayLike, times: ArrayLike) -> NDArray[np.float64]:
    """S(t) read off the Kaplan-Meier curve at the requested times (step interpolation)."""
    curve = kaplan_meier(durations, events)
    knots = curve["time"].to_numpy()
    values = curve["survival"].to_numpy()
    query = np.asarray(times, dtype=float)
    idx = np.searchsorted(knots, query, side="right") - 1
    return np.asarray(np.where(idx < 0, 1.0, values[np.clip(idx, 0, values.size - 1)]), dtype=float)


def median_survival(durations: ArrayLike, events: ArrayLike) -> float:
    """First time the survival curve crosses 0.5 — typical lifetime; NaN if it never does.

    Prefer this to mean tenure when churn is skewed (it always is); NaN itself is informative:
    most of the base outlives the observation window.
    """
    curve = kaplan_meier(durations, events)
    below = curve.filter(pl.col("survival") <= 0.5)
    return float(below["time"][0]) if below.height else float("nan")


def restricted_mean_survival(
    durations: ArrayLike, events: ArrayLike, *, horizon: float | None = None
) -> float:
    """Expected survival time up to ``horizon`` — area under the KM curve (RMST).

    The CLV-ready number: expected retained periods per customer over the horizon; multiply by
    margin per period. Always estimable (unlike the unrestricted mean under censoring) and the
    honest way to compare retention between cohorts.
    """
    curve = kaplan_meier(durations, events)
    knots = curve["time"].to_numpy()
    limit = float(horizon) if horizon is not None else float(knots.max())
    grid = np.concatenate([[0.0], knots[knots <= limit], [limit]])
    grid = np.unique(grid)
    steps = survival_at(durations, events, grid[:-1])  # S is right-continuous: left value rules
    return float(np.sum(steps * np.diff(grid)))


def cox_ph(
    df: pl.DataFrame, *, duration: str, event: str, x: Sequence[str], confidence: float = 0.95
) -> pl.DataFrame:
    """Cox proportional-hazards regression: covariates → hazard ratios with CIs.

    ``hazard_ratio`` = exp(coef): 1.3 on support_tickets means each extra ticket raises the
    churn hazard 30% *at every tenure*, holding the rest fixed. That "at every tenure" is the
    proportional-hazards assumption — check it by fitting on early/late tenure splits and
    comparing. Semi-parametric: no baseline-hazard shape assumed.
    """
    from scipy.stats import norm
    from statsmodels.duration.hazard_regression import PHReg

    t, e = _durations(df[duration].to_numpy(), df[event].to_numpy())
    exog = df.select(x).to_pandas()
    result = PHReg(t, exog, status=e).fit()
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    coef = np.asarray(result.params, dtype=float)
    se = np.asarray(result.bse, dtype=float)
    return pl.DataFrame(
        {
            "term": list(x),
            "coef": coef,
            "hazard_ratio": np.exp(coef),
            "std_err": se,
            "p_value": np.asarray(result.pvalues, dtype=float),
            "hr_ci_low": np.exp(coef - z * se),
            "hr_ci_high": np.exp(coef + z * se),
        }
    ).sort(pl.col("p_value"))
