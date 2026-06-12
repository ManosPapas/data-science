"""Regression for *inference* — read effects with SEs/p-values, and check the assumptions.

``modeling`` predicts; this module explains. Fitters return a :class:`FitSummary` (coefficient
table + model stats): :func:`ols_fit` (continuous y), :func:`glm_fit` (counts, binary, skewed
amounts), :func:`fixed_effects` (panel data — absorb entity-level confounding),
:func:`mixed_effects` (random intercepts — partial pooling). Inference is only as good as the
assumptions, so the diagnostics live alongside: :func:`vif` (collinearity), :func:`breusch_pagan`
(heteroscedasticity), :func:`durbin_watson` (autocorrelation), and residual normality via
``stats.normality_test`` / ``viz.eda.qq``; ``linear_assumptions`` runs the lot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray

from core.analytics.stats import Floats, TestResult, normality_test


def _exog(features: pl.DataFrame | Floats) -> Any:
    """Feature matrix with an intercept column prepended (what statsmodels tests expect)."""
    x = features.to_numpy() if isinstance(features, pl.DataFrame) else np.asarray(features, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(x.shape[0]), x])


# --- Fitting (inference) --------------------------------------------------------------------


@dataclass(frozen=True)
class FitSummary:
    """A fitted model read for inference: coefficient table plus model-level stats."""

    coefficients: pl.DataFrame  # term / coef / std_err / statistic / p_value / ci_low / ci_high
    r_squared: float | None
    aic: float
    n: int
    group_variance: float | None = None


def _coef_frame(
    terms: Sequence[str],
    coefs: NDArray[np.float64],
    std_errs: NDArray[np.float64],
    statistics: NDArray[np.float64],
    p_values: NDArray[np.float64],
    ci_low: NDArray[np.float64],
    ci_high: NDArray[np.float64],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "term": list(terms),
            "coef": coefs.astype(float),
            "std_err": std_errs.astype(float),
            "statistic": statistics.astype(float),
            "p_value": p_values.astype(float),
            "ci_low": ci_low.astype(float),
            "ci_high": ci_high.astype(float),
        }
    )


def ols_fit(df: pl.DataFrame, *, y: str, x: Sequence[str]) -> FitSummary:
    """OLS with inference — for *reading* effects (coef, SE, t, p, CI), not for predicting.

    Each coefficient is the expected change in y per unit of that feature, the others held
    fixed. The p-values and CIs are only as honest as the assumptions — run
    :func:`linear_assumptions` on the residuals before quoting them. Test moderation by adding
    product columns first (``transform.add_interactions``).
    """
    import statsmodels.api as sm

    frame = df.select([y, *x]).drop_nulls()
    result = sm.OLS(frame[y].to_numpy(), sm.add_constant(frame.select(x).to_numpy())).fit()
    ci = np.asarray(result.conf_int(), dtype=float)
    table = _coef_frame(
        ["intercept", *x],
        np.asarray(result.params, dtype=float),
        np.asarray(result.bse, dtype=float),
        np.asarray(result.tvalues, dtype=float),
        np.asarray(result.pvalues, dtype=float),
        ci[:, 0],
        ci[:, 1],
    )
    return FitSummary(table, float(result.rsquared), float(result.aic), frame.height)


def glm_fit(df: pl.DataFrame, *, y: str, x: Sequence[str], family: str = "poisson") -> FitSummary:
    """Generalized linear model — the right likelihood and link when y isn't normal.

    Families: 'poisson' for counts (log link — exp(coef) reads as a rate multiplier), 'binomial'
    for 0/1 outcomes (logit — coefs are log-odds), 'gamma' for positive skewed amounts (log
    link), 'gaussian' = OLS. Statistics are Wald z; compare families/feature sets by AIC.
    Prediction-oriented twins live in the registry ('poisson', 'gamma', 'tweedie', 'logistic').
    """
    import statsmodels.api as sm

    families = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(sm.families.links.Log()),
    }
    if family not in families:
        raise ValueError(f"unknown family: {family!r} (choose from {sorted(families)})")
    frame = df.select([y, *x]).drop_nulls()
    result = sm.GLM(
        frame[y].to_numpy(), sm.add_constant(frame.select(x).to_numpy()), family=families[family]
    ).fit()
    ci = np.asarray(result.conf_int(), dtype=float)
    table = _coef_frame(
        ["intercept", *x],
        np.asarray(result.params, dtype=float),
        np.asarray(result.bse, dtype=float),
        np.asarray(result.tvalues, dtype=float),
        np.asarray(result.pvalues, dtype=float),
        ci[:, 0],
        ci[:, 1],
    )
    return FitSummary(table, None, float(result.aic), frame.height)


def fixed_effects(df: pl.DataFrame, *, y: str, x: Sequence[str], entity: str) -> FitSummary:
    """Within-entity (fixed-effects) regression — absorbs all time-invariant entity confounding.

    Demeans y and x inside each entity so only *within*-entity variation identifies the slopes:
    anything constant per entity (location, brand, baseline quality) cancels — the panel-data
    workhorse. SEs and p-values use the demeaning-corrected dof (n - k - #entities);
    ``r_squared`` is the within-R². Entity-constant features are unidentifiable here — use
    :func:`mixed_effects` if you need them.
    """
    import statsmodels.api as sm
    from scipy.stats import t as t_dist

    frame = df.select([y, *x, entity]).drop_nulls()
    demeaned = frame.with_columns(
        [(pl.col(col) - pl.col(col).mean().over(entity)).alias(col) for col in (y, *x)]
    )
    result = sm.OLS(demeaned[y].to_numpy(), demeaned.select(x).to_numpy()).fit()
    dof = frame.height - len(x) - frame[entity].n_unique()
    if dof <= 0:
        raise ValueError("not enough observations per entity for fixed effects")
    coefs = np.asarray(result.params, dtype=float)
    std_errs = np.asarray(result.bse, dtype=float) * float(np.sqrt(result.df_resid / dof))
    t_stats = coefs / std_errs
    p_values = 2.0 * np.asarray(t_dist.sf(np.abs(t_stats), dof), dtype=float)
    half = float(t_dist.ppf(0.975, dof)) * std_errs
    table = _coef_frame(list(x), coefs, std_errs, t_stats, p_values, coefs - half, coefs + half)
    return FitSummary(table, float(result.rsquared), float(result.aic), frame.height)


def mixed_effects(df: pl.DataFrame, *, y: str, x: Sequence[str], group: str) -> FitSummary:
    """Random-intercept mixed model — partial pooling across groups (stores, users, regions).

    Fixed slopes for ``x`` plus a per-group intercept drawn from a fitted normal: small groups
    borrow strength from the rest instead of getting their own noisy dummy (the regression
    cousin of ``bayes.hierarchical_rates``). ``group_variance`` is the between-group intercept
    variance — compare it to the residual variance to see how much heterogeneity the grouping
    carries. Wald z inference; wants a decent number of groups (≳ 8-10).
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    frame = df.select([y, *x, group]).drop_nulls()
    result = sm.MixedLM(
        frame[y].to_numpy(),
        sm.add_constant(frame.select(x).to_numpy()),
        groups=frame[group].to_numpy(),
    ).fit(reml=True)
    coefs = np.asarray(result.fe_params, dtype=float)
    std_errs = np.asarray(result.bse_fe, dtype=float)
    z_stats = coefs / std_errs
    p_values = 2.0 * np.asarray(norm.sf(np.abs(z_stats)), dtype=float)
    half = float(norm.ppf(0.975)) * std_errs
    table = _coef_frame(
        ["intercept", *x], coefs, std_errs, z_stats, p_values, coefs - half, coefs + half
    )
    try:
        aic = float(result.aic)
    except (AttributeError, TypeError):
        aic = float("nan")
    return FitSummary(
        table,
        None,
        aic,
        frame.height,
        group_variance=float(np.asarray(result.cov_re)[0, 0]),
    )


# --- Assumption diagnostics -----------------------------------------------------------------


def vif(df: pl.DataFrame, *, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Variance inflation factor per numeric feature — how collinear is it with the others?

    VIF_i = 1 / (1 - R²) of feature i regressed on the rest: 1 = independent, > 5 worrying,
    > 10 unstable coefficients (drop/combine features, or switch to Ridge/Lasso/PCA).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    numeric = (df.select(cs.numeric()) if columns is None else df.select(columns)).drop_nulls()
    names = numeric.columns
    exog = _exog(numeric)
    values = [float(variance_inflation_factor(exog, i + 1)) for i in range(len(names))]
    return pl.DataFrame({"feature": names, "vif": values}).sort("vif", descending=True)


def breusch_pagan(residuals: Floats, features: pl.DataFrame | Floats) -> TestResult:
    """Breusch-Pagan test. H0: residual variance is constant (homoscedastic).

    A small p-value means the error variance moves with the features — OLS standard errors and
    p-values are then unreliable (use robust/HC errors, transform y, or model the variance).
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    statistic, p_value, _, _ = het_breuschpagan(np.asarray(residuals, float), _exog(features))
    return TestResult(float(statistic), float(p_value))


def durbin_watson(residuals: Floats) -> float:
    """Durbin-Watson statistic on row-ordered residuals — first-order autocorrelation.

    Reads ~2 = independent, < 1.5 positive autocorrelation, > 2.5 negative. Order the residuals by
    time before calling; with autocorrelation, naive standard errors overstate the evidence.
    """
    from statsmodels.stats.stattools import durbin_watson as _durbin_watson

    return float(_durbin_watson(np.asarray(residuals, dtype=float)))


def linear_assumptions(features: pl.DataFrame, residuals: Floats) -> dict[str, float | str]:
    """One-stop OLS assumption check: normality, homoscedasticity, autocorrelation, collinearity.

    Pass the (numeric) training features and the fitted model's residuals in row/time order.
    Healthy: ``normality_p`` and ``homoscedasticity_p`` above 0.05, ``durbin_watson`` near 2,
    ``max_vif`` below 5. Linearity itself is visual — see ``viz.model.residuals``.
    """
    collinearity = vif(features)
    return {
        "normality_p": normality_test(residuals).p_value,
        "homoscedasticity_p": breusch_pagan(residuals, features).p_value,
        "durbin_watson": durbin_watson(residuals),
        "max_vif": float(collinearity["vif"][0]),
        "max_vif_feature": str(collinearity["feature"][0]),
    }
