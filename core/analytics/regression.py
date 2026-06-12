"""Linear-model assumption diagnostics — multicollinearity, heteroscedasticity, autocorrelation.

OLS predictions survive mild violations; *inference* (coefficients, CIs, p-values) does not. Check
residual normality with ``stats.normality_test`` / ``viz.eda.qq``, then the rest here:
:func:`vif` (are predictors collinear?), :func:`breusch_pagan` (is residual variance constant?),
:func:`durbin_watson` (are consecutive residuals correlated?). ``linear_assumptions`` runs the lot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs

from core.analytics.stats import Floats, TestResult, normality_test


def _exog(features: pl.DataFrame | Floats) -> Any:
    """Feature matrix with an intercept column prepended (what statsmodels tests expect)."""
    x = features.to_numpy() if isinstance(features, pl.DataFrame) else np.asarray(features, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(x.shape[0]), x])


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
