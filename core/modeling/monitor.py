"""Model & data drift monitoring — is fresh data still the data the model was trained on?

PSI (population stability index) over baseline-quantile bins plus the two-sample KS test, per
column. Standard operating read on PSI: < 0.1 stable, 0.1-0.2 drifting, > 0.2 drifted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


def label_drift(expected: ArrayLike, actual: ArrayLike) -> TestResult:
    """Chi-square check that the class mix has not shifted (PSI/KS cover numeric columns only).

    Pass the baseline and fresh *label* arrays — true labels when you have them (prior shift), or
    predicted classes as the early proxy. A small p-value means the class proportions moved:
    re-examine thresholds and consider retraining. Labels unseen in the baseline get a tiny floor
    share so a brand-new class registers as drift instead of crashing the test.
    """
    from scipy.stats import chisquare

    e = np.asarray(expected).astype(str)
    a = np.asarray(actual).astype(str)
    labels = np.unique(np.concatenate([e, a]))
    expected_counts = np.array([(e == label).sum() for label in labels], dtype=float)
    actual_counts = np.array([(a == label).sum() for label in labels], dtype=float)
    shares = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    shares = shares / shares.sum()
    result = chisquare(actual_counts, f_exp=actual_counts.sum() * shares)
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


@dataclass(frozen=True)
class ControlLimits:
    """Shewhart control band derived from a stable baseline period."""

    center: float
    lower: float
    upper: float


def control_limits(baseline: ArrayLike, *, sigmas: float = 3.0) -> ControlLimits:
    """Mean ± k·sigma limits from an in-control baseline — the classic process-watch band.

    Points outside the band are signals, not noise (at 3-sigma, ~0.3% false alarms on a stable
    normal process). Compute the limits on a period you *trust*, then judge fresh data against
    them — recomputing limits on drifting data hides the drift.
    """
    x = np.asarray(baseline, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 baseline points")
    center = float(x.mean())
    spread = float(x.std(ddof=1))
    return ControlLimits(center, center - sigmas * spread, center + sigmas * spread)


def ewma_alerts(
    values: ArrayLike,
    baseline: ArrayLike,
    *,
    lam: float = 0.2,
    sigmas: float = 3.0,
) -> pl.DataFrame:
    """EWMA control chart — the early-warning system for slow drifts a Shewhart band misses.

    Each point is an exponentially weighted average (memory ``lam``: smaller = longer memory =
    more sensitive to small persistent shifts); limits widen to their asymptote as the window
    fills. ``alert`` marks the first symptoms of a metric quietly walking away from baseline —
    fire a drift investigation (``drift_report``) when alerts persist.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    base = np.asarray(baseline, dtype=float)
    if base.size < 2:
        raise ValueError("need at least 2 baseline points")
    x = np.asarray(values, dtype=float)
    center = float(base.mean())
    spread = float(base.std(ddof=1))
    ewma = np.empty(x.size)
    level = center
    for i, value in enumerate(x):
        level = lam * value + (1.0 - lam) * level
        ewma[i] = level
    steps = np.arange(1, x.size + 1)
    width = sigmas * spread * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2.0 * steps)))
    lower, upper = center - width, center + width
    return pl.DataFrame(
        {
            "t": np.arange(x.size),
            "value": x,
            "ewma": ewma,
            "lower": lower,
            "upper": upper,
            "alert": (ewma < lower) | (ewma > upper),
        }
    )
