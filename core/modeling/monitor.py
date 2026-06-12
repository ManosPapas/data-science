"""Model & data drift monitoring — is fresh data still the data the model was trained on?

PSI (population stability index) over baseline-quantile bins plus the two-sample KS test, per
column. Standard operating read on PSI: < 0.1 stable, 0.1-0.2 drifting, > 0.2 drifted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import ArrayLike

from core.analytics.stats import TestResult


def psi(expected: ArrayLike, actual: ArrayLike, *, bins: int = 10) -> float:
    """Population stability index between a baseline sample and a fresh one.

    Bins are quantiles of the *baseline* distribution; proportions are floored so an empty bin
    can't produce infinities. Raises when the baseline has too little variation to bin.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        raise ValueError("baseline has too little variation to bin — PSI is not meaningful")
    expected_pct = np.clip(np.histogram(e, bins=edges)[0] / e.size, 1e-6, None)
    actual_pct = np.clip(np.histogram(a, bins=edges)[0] / a.size, 1e-6, None)
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_drift(expected: ArrayLike, actual: ArrayLike) -> TestResult:
    """Two-sample Kolmogorov-Smirnov test (small p-value = the distributions differ)."""
    from scipy.stats import ks_2samp

    result = ks_2samp(np.asarray(expected, dtype=float), np.asarray(actual, dtype=float))
    return TestResult(float(result.statistic), float(result.pvalue))


def drift_report(
    baseline: pl.DataFrame,
    current: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    psi_threshold: float = 0.2,
) -> pl.DataFrame:
    """Per-column PSI + KS for ``columns`` (default: numeric columns present in both frames).

    Run it on features *and* on the model's scores — score drift is the early-warning signal.
    """
    if columns is None:
        in_baseline = set(baseline.select(cs.numeric()).columns)
        columns = [col for col in current.select(cs.numeric()).columns if col in in_baseline]
    rows = []
    for col in columns:
        e = baseline[col].drop_nulls().to_numpy()
        a = current[col].drop_nulls().to_numpy()
        stability = psi(e, a)
        ks = ks_drift(e, a)
        rows.append(
            {
                "column": col,
                "psi": round(stability, 4),
                "ks_stat": round(ks.statistic, 4),
                "ks_p": ks.p_value,
                "drifted": stability > psi_threshold,
            }
        )
    return pl.DataFrame(rows).sort("psi", descending=True)
