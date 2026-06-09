"""Statistics for EDA and decision-making: summaries, missingness, correlation, tests, CIs.

These wrap polars/numpy/scipy with names and return types that read clearly at the call site, so a
notebook says ``welch_t_test(a, b).p_value`` rather than re-deriving the plumbing each time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from scipy import stats

Floats = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class TestResult:
    """A test statistic and its p-value."""

    statistic: float
    p_value: float


def summary(df: pl.DataFrame) -> pl.DataFrame:
    """One row per column: dtype, count, nulls, n_unique, and numeric stats where applicable."""
    height = df.height
    rows: list[dict[str, Any]] = []
    for name, dtype in df.schema.items():
        series = df[name]
        n_null = series.null_count()
        numeric = dtype.is_numeric()
        rows.append(
            {
                "column": name,
                "dtype": str(dtype),
                "count": height - n_null,
                "n_null": n_null,
                "pct_null": round(n_null / height * 100, 2) if height else 0.0,
                "n_unique": series.n_unique(),
                "mean": _stat(series.mean()) if numeric else None,
                "std": _stat(series.std()) if numeric else None,
                "min": _stat(series.min()) if numeric else None,
                "q25": _stat(series.quantile(0.25)) if numeric else None,
                "median": _stat(series.median()) if numeric else None,
                "q75": _stat(series.quantile(0.75)) if numeric else None,
                "max": _stat(series.max()) if numeric else None,
            }
        )
    return pl.DataFrame(rows)


def _stat(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def cardinality(df: pl.DataFrame) -> pl.DataFrame:
    """Per-column distinct count and its share of rows (spot IDs, pick categoricals)."""
    height = df.height
    names = df.columns
    counts = [df[name].n_unique() for name in names]
    pct = [round(count / height * 100, 2) if height else 0.0 for count in counts]
    return pl.DataFrame({"column": names, "n_unique": counts, "pct_unique": pct}).sort(
        "n_unique", descending=True
    )


def missingness(df: pl.DataFrame) -> pl.DataFrame:
    """Per-column null count and percentage, most-missing first."""
    height = df.height
    null_counts = df.null_count()
    columns = df.columns
    counts = [int(null_counts[col][0]) for col in columns]
    pct = [round(count / height * 100, 2) if height else 0.0 for count in counts]
    return pl.DataFrame({"column": columns, "n_null": counts, "pct_null": pct}).sort(
        "n_null", descending=True
    )


def correlation(df: pl.DataFrame) -> pl.DataFrame:
    """Pearson correlation across numeric columns (rows containing nulls are dropped)."""
    numeric = df.select(cs.numeric()).drop_nulls()
    names = numeric.columns
    if len(names) < 2:
        return pl.DataFrame()
    matrix = np.corrcoef(numeric.to_numpy(), rowvar=False)
    out = pl.DataFrame({"column": names})
    for index, name in enumerate(names):
        out = out.with_columns(pl.Series(name, matrix[:, index]))
    return out


def pct_change(current: float, previous: float) -> float | None:
    """Relative change from ``previous`` to ``current``; ``None`` if the base is zero."""
    if previous == 0:
        return None
    return round((current - previous) / previous, 4)


def cohens_d(a: Floats, b: Floats) -> float:
    """Standardized mean difference (pooled SD) — effect size for two samples."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def welch_t_test(a: Floats, b: Floats) -> TestResult:
    """Welch's t-test (unequal variances) — does the mean differ between two samples?"""
    statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
    return TestResult(float(statistic), float(p_value))


def mann_whitney(a: Floats, b: Floats) -> TestResult:
    """Mann-Whitney U — non-parametric alternative when the t-test's assumptions don't hold."""
    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    return TestResult(float(statistic), float(p_value))


def mean_confidence_interval(data: Floats, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the given confidence level using the t distribution."""
    arr = np.asarray(data, dtype=float)
    mean = float(arr.mean())
    half_width = float(stats.sem(arr)) * float(stats.t.ppf((1 + confidence) / 2, len(arr) - 1))
    return mean, mean - half_width, mean + half_width
