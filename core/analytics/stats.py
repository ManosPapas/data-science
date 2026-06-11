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
from numpy.typing import ArrayLike, NDArray
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


# --- Distribution & shape -----------------------------------------------------------------------


def describe_distribution(data: Floats) -> dict[str, float]:
    """Shape of a 1-D sample: mean, std, skew, kurtosis, and 5/25/50/75/95 percentiles."""
    arr = np.asarray(data, dtype=float)
    percentiles = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "skew": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def normality_test(data: Floats, *, method: str = "shapiro") -> TestResult:
    """Test whether a sample is normally distributed ('shapiro' or 'dagostino')."""
    arr = np.asarray(data, dtype=float)
    statistic, p_value = stats.normaltest(arr) if method == "dagostino" else stats.shapiro(arr)
    return TestResult(float(statistic), float(p_value))


def outlier_bounds(
    data: Floats, *, method: str = "iqr", factor: float = 1.5
) -> tuple[float, float]:
    """Lower/upper outlier bounds via IQR (factor * IQR) or z-score (factor * std)."""
    arr = np.asarray(data, dtype=float)
    if method == "zscore":
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        return mean - factor * std, mean + factor * std
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


# --- Effect sizes -------------------------------------------------------------------------------


def hedges_g(a: Floats, b: Floats) -> float:
    """Bias-corrected standardized mean difference (Cohen's d adjusted for small samples)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    correction = 1 - 3 / (4 * (len(x) + len(y)) - 9)
    return cohens_d(x, y) * correction


def cliffs_delta(a: Floats, b: Floats) -> float:
    """Non-parametric effect size in [-1, 1]: P(a > b) - P(a < b)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sign(np.subtract.outer(x, y)).sum() / (x.size * y.size))


def eta_squared(groups: Sequence[Floats]) -> float:
    """Proportion of variance explained by group membership (ANOVA effect size)."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    grand = np.concatenate(arrays)
    ss_total = float(((grand - grand.mean()) ** 2).sum())
    ss_between = float(sum(a.size * (a.mean() - grand.mean()) ** 2 for a in arrays))
    return ss_between / ss_total if ss_total else 0.0


# --- Hypothesis tests ---------------------------------------------------------------------------


def _contingency(a: ArrayLike, b: ArrayLike) -> NDArray[np.int_]:
    x, y = np.asarray(a), np.asarray(b)
    cats_y = np.unique(y)
    rows = [[int(np.sum((x == cx) & (y == cy))) for cy in cats_y] for cx in np.unique(x)]
    return np.asarray(rows)


def anova(*groups: Floats) -> TestResult:
    """One-way ANOVA across 2+ groups — do the means differ?"""
    statistic, p_value = stats.f_oneway(*[np.asarray(g, dtype=float) for g in groups])
    return TestResult(float(statistic), float(p_value))


def kruskal(*groups: Floats) -> TestResult:
    """Kruskal-Wallis — non-parametric ANOVA across 2+ groups."""
    statistic, p_value = stats.kruskal(*[np.asarray(g, dtype=float) for g in groups])
    return TestResult(float(statistic), float(p_value))


def chi_square(a: ArrayLike, b: ArrayLike) -> TestResult:
    """Chi-square test of independence between two categorical variables."""
    statistic, p_value, _, _ = stats.chi2_contingency(_contingency(a, b))
    return TestResult(float(statistic), float(p_value))


def proportions_test(successes: Sequence[int], totals: Sequence[int]) -> TestResult:
    """Two-proportion z-test (e.g. conversion A vs B)."""
    from statsmodels.stats.proportion import proportions_ztest

    statistic, p_value = proportions_ztest(np.asarray(successes), np.asarray(totals))
    return TestResult(float(statistic), float(p_value))


def correlation_test(a: Floats, b: Floats, *, method: str = "pearson") -> TestResult:
    """Correlation with a p-value ('pearson' or 'spearman')."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    statistic, p_value = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
    return TestResult(float(statistic), float(p_value))


# --- Relationships ------------------------------------------------------------------------------


def spearman(df: pl.DataFrame) -> pl.DataFrame:
    """Spearman rank correlation across numeric columns."""
    numeric = df.select(cs.numeric()).drop_nulls()
    names = numeric.columns
    if len(names) < 2:
        return pl.DataFrame()
    ranked = np.apply_along_axis(stats.rankdata, 0, numeric.to_numpy())
    matrix = np.corrcoef(ranked, rowvar=False)
    out = pl.DataFrame({"column": names})
    for index, name in enumerate(names):
        out = out.with_columns(pl.Series(name, matrix[:, index]))
    return out


def mutual_information(df: pl.DataFrame, target: str, *, task: str = "regression") -> pl.DataFrame:
    """Mutual information between each numeric feature and the target (feature relevance)."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    feature_cols = [c for c in df.select(cs.numeric()).columns if c != target]
    combined = df.select([*feature_cols, target]).drop_nulls()
    estimator = mutual_info_classif if task == "classification" else mutual_info_regression
    scores = estimator(combined.select(feature_cols).to_numpy(), combined[target].to_numpy())
    return pl.DataFrame({"feature": feature_cols, "mi": scores}).sort("mi", descending=True)


# --- Size & power -------------------------------------------------------------------------------


def sample_size_mean(effect_size: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Per-group n to detect a standardized mean difference (Cohen's d) at the target power."""
    from statsmodels.stats.power import TTestIndPower

    needed = TTestIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha)
    return int(np.ceil(needed))


def sample_size_proportion(p1: float, p2: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Per-group n to detect a difference between two proportions."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    effect = proportion_effectsize(p1, p2)
    needed = NormalIndPower().solve_power(effect_size=effect, power=power, alpha=alpha)
    return int(np.ceil(abs(needed)))


def power(effect_size: float, n: int, *, alpha: float = 0.05) -> float:
    """Power of a two-sample t-test for a given effect size and per-group n."""
    from statsmodels.stats.power import TTestIndPower

    return float(TTestIndPower().power(effect_size=effect_size, nobs1=n, alpha=alpha))


# --- Compare segments ---------------------------------------------------------------------------


def group_summary(df: pl.DataFrame, value: str, group: str) -> pl.DataFrame:
    """Per-group count, mean, std, and 95% CI half-width for ``value``."""
    summarized = df.group_by(group).agg(
        pl.len().alias("n"),
        pl.col(value).mean().alias("mean"),
        pl.col(value).std().alias("std"),
    )
    return summarized.with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci95")).sort(
        group
    )


def compare_groups(df: pl.DataFrame, value: str, group: str) -> dict[str, Any]:
    """Compare ``value`` across the levels of ``group``: auto-pick the test + report an effect size.

    Two levels -> Welch t (or Mann-Whitney if non-normal); 3+ -> ANOVA (or Kruskal).
    """
    levels = df.select(group).unique().to_series().to_list()
    samples = [df.filter(pl.col(group) == level)[value].drop_nulls().to_numpy() for level in levels]
    normal = all(normality_test(s).p_value > 0.05 for s in samples if len(s) >= 3)
    if len(samples) == 2:
        result = (
            welch_t_test(samples[0], samples[1]) if normal else mann_whitney(samples[0], samples[1])
        )
        test = "welch_t" if normal else "mann_whitney"
        effect = cohens_d(samples[0], samples[1])
    else:
        result = anova(*samples) if normal else kruskal(*samples)
        test = "anova" if normal else "kruskal"
        effect = eta_squared(samples)
    return {
        "test": test,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "effect_size": effect,
    }
