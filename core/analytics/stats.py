"""Statistics for EDA and decision-making: summaries, missingness, correlation, tests, CIs.

These wrap polars/numpy/scipy with names and return types that read clearly at the call site, so a
notebook says ``welch_t_test(a, b).p_value`` rather than re-deriving the plumbing each time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def missingness_dependence(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Does ``column``'s missingness depend on other columns? (MCAR vs MAR triage.)

    Compares every other column between rows where ``column`` is null and rows where it isn't —
    Welch t for numeric, chi-square otherwise. Small p-values mean the data are missing *at random
    given that column* (MAR) rather than completely at random (MCAR), so dropping rows or
    mean-imputing would bias the analysis; impute conditionally or add a missing flag instead.
    """
    schema = {
        "column": pl.String,
        "test": pl.String,
        "statistic": pl.Float64,
        "p_value": pl.Float64,
    }
    flag = df[column].is_null()
    rows: list[dict[str, Any]] = []
    for other, dtype in df.schema.items():
        if other == column:
            continue
        keep = df[other].is_not_null()
        values, groups = df[other].filter(keep), flag.filter(keep)
        n_missing = int(groups.sum())
        if n_missing < 2 or groups.len() - n_missing < 2:
            continue
        if dtype.is_numeric():
            test = "welch_t"
            result = welch_t_test(
                values.filter(groups).to_numpy(), values.filter(~groups).to_numpy()
            )
        else:
            test = "chi_square"
            result = chi_square(values.cast(pl.String).to_numpy(), groups.to_numpy())
        rows.append(
            {
                "column": other,
                "test": test,
                "statistic": result.statistic,
                "p_value": result.p_value,
            }
        )
    return pl.DataFrame(rows, schema=schema).sort("p_value")


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


def proportion_confidence_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float, float]:
    """(rate, lower, upper) Wilson score interval for a proportion (conversion, CTR, churn).

    Stays inside [0, 1] and behaves at small n and extreme rates, where the naive Wald interval
    (p ± z·SE) collapses to zero width or spills outside — the standard error bar for rates.
    """
    z = float(stats.norm.ppf((1 + confidence) / 2))
    rate = successes / trials
    denom = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denom
    half = z * float(np.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))) / denom
    return rate, centre - half, centre + half


def bootstrap_ci(
    data: Floats,
    statistic: Callable[..., Any] = np.mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    method: str = "BCa",
    seed: int = 42,
) -> tuple[float, float, float]:
    """(estimate, lower, upper) for *any* statistic by resampling — no formula, no normality.

    Resamples the data with replacement, recomputes ``statistic`` each time, and reads the CI off
    that empirical distribution ('BCa' corrects bias and skew; 'percentile'/'basic' are simpler).
    The tool for medians, ratios, quantiles, trimmed means — anything without a clean closed-form
    standard error. Mirrors the CLT idea by simulation instead of theory.
    """
    arr = np.asarray(data, dtype=float)
    result = stats.bootstrap(
        (arr,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method=method,
        rng=np.random.default_rng(seed),
    )
    interval = result.confidence_interval
    return float(statistic(arr)), float(interval.low), float(interval.high)


def bayes_rule(prior: float, true_positive_rate: float, false_positive_rate: float) -> float:
    """Posterior P(hypothesis | positive signal) via Bayes' theorem.

    ``prior`` = P(H) base rate, ``true_positive_rate`` = P(signal | H), ``false_positive_rate``
    = P(signal | not H). The base-rate lesson: a 99%-sensitive test with a 5% false-positive
    rate on a 1% prior yields only ~17% — evidence *updates* the prior, it doesn't replace it.
    Useful for fraud alerts, test results, lead scoring; the same arithmetic underlies
    ``analytics.bayes`` and the ``experiment.bayes_*`` tools.
    """
    signal_rate = true_positive_rate * prior + false_positive_rate * (1.0 - prior)
    return true_positive_rate * prior / signal_rate if signal_rate else 0.0


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
    """Test whether a sample is normal ('shapiro', 'dagostino', or 'ks').

    'ks' is the Lilliefors-corrected Kolmogorov-Smirnov test (plain KS is anticonservative when
    mean/std are estimated from the same sample). Shapiro-Wilk has the best power below ~5k rows.
    """
    arr = np.asarray(data, dtype=float)
    if method == "ks":
        from statsmodels.stats.diagnostic import lilliefors

        statistic, p_value = lilliefors(arr, dist="norm")
    elif method == "dagostino":
        statistic, p_value = stats.normaltest(arr)
    else:
        statistic, p_value = stats.shapiro(arr)
    return TestResult(float(statistic), float(p_value))


def fit_distribution(data: Floats, dist: str = "norm") -> dict[str, Any]:
    """Fit a scipy distribution by maximum likelihood; params, log-likelihood, AIC, and KS fit.

    ``dist`` is any scipy.stats name ('norm', 'lognorm', 'expon', 'gamma', 'weibull_min', ...).
    ``params`` come back in scipy order ``(shape..., loc, scale)`` and plug straight into
    ``scipy.stats.<dist>(*params)``. A small ``ks_p`` says the fitted shape still mismatches the
    sample; compare candidates with :func:`best_distribution`.
    """
    arr = np.asarray(data, dtype=float)
    model = getattr(stats, dist)
    params = model.fit(arr)
    loglik = float(np.sum(model.logpdf(arr, *params)))
    ks_stat, ks_p = stats.kstest(arr, dist, args=params)
    return {
        "dist": dist,
        "params": tuple(float(p) for p in params),
        "loglik": loglik,
        "aic": 2.0 * len(params) - 2.0 * loglik,
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def best_distribution(
    data: Floats,
    candidates: Sequence[str] = ("norm", "lognorm", "expon", "gamma", "weibull_min"),
) -> pl.DataFrame:
    """MLE-fit each candidate distribution and rank by AIC (best first)."""
    fits = [fit_distribution(data, dist) for dist in candidates]
    return pl.DataFrame([{k: v for k, v in f.items() if k != "params"} for f in fits]).sort("aic")


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


def permutation_test(
    a: Floats, b: Floats, *, n_resamples: int = 10_000, seed: int = 42
) -> TestResult:
    """Difference in means with significance from label shuffling — no distributional assumptions.

    Under H0 the group labels are arbitrary, so reshuffling them maps out the null distribution
    of mean(a) - mean(b); the p-value is the share of shuffles at least as extreme as observed.
    Reach for it at small n or with ugly distributions when you still want to compare *means*
    (``mann_whitney`` compares ranks, a different question).
    """

    def _mean_diff(x: Any, y: Any, axis: int = -1) -> Any:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = stats.permutation_test(
        (np.asarray(a, dtype=float), np.asarray(b, dtype=float)),
        _mean_diff,
        permutation_type="independent",
        n_resamples=n_resamples,
        alternative="two-sided",
        rng=np.random.default_rng(seed),
    )
    return TestResult(float(result.statistic), float(result.pvalue))


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


def simpsons_check(df: pl.DataFrame, *, x: str, y: str, group: str) -> dict[str, Any]:
    """Does the x-y association flip once you condition on ``group``? (Simpson's paradox.)

    Compares the pooled OLS slope of y on x against per-group slopes. ``reversal=True`` — the
    pooled and size-weighted within-group slopes disagree in sign — means the headline trend is
    an artifact of group composition. Which margin to report is a causal question: condition on
    the group if it's a confounder, don't if it's on the causal path.
    """
    sub = df.select([x, y, group]).drop_nulls()

    def _slope(frame: pl.DataFrame) -> float | None:
        xs = frame[x].to_numpy().astype(float)
        ys = frame[y].to_numpy().astype(float)
        if xs.size < 3 or float(xs.var(ddof=1)) == 0.0:
            return None
        return float(np.cov(xs, ys)[0, 1] / xs.var(ddof=1))

    overall = _slope(sub)
    rows: list[dict[str, Any]] = []
    for level in sub.select(group).unique().sort(group).to_series().to_list():
        slope = _slope(sub.filter(pl.col(group) == level))
        if slope is not None:
            rows.append(
                {"group": level, "n": sub.filter(pl.col(group) == level).height, "slope": slope}
            )
    by_group = pl.DataFrame(
        rows, schema={"group": sub.schema[group], "n": pl.Int64, "slope": pl.Float64}
    )
    within = (
        float(np.average(by_group["slope"].to_numpy(), weights=by_group["n"].to_numpy()))
        if by_group.height
        else None
    )
    reversal = overall is not None and within is not None and overall * within < 0
    return {
        "overall_slope": overall,
        "within_slope": within,
        "by_group": by_group,
        "reversal": reversal,
    }


# --- Information theory ---------------------------------------------------------------------


def entropy(labels: ArrayLike, *, base: float = 2.0) -> float:
    """Shannon entropy of a categorical sample (bits by default) — how unpredictable is it?

    0 = a single category; log2(k) = uniform over k categories. The uncertainty currency that
    information gain, mutual information, and tree-split criteria (entropy vs Gini) trade in.
    """
    _, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return float(stats.entropy(counts, base=base))


def kl_divergence(p: Floats, q: Floats, *, base: float = 2.0) -> float:
    """KL(P ‖ Q): the extra bits paid for modelling P with Q. Asymmetric; 0 iff identical.

    Takes two probability vectors over the same categories (normalized internally). Relatives:
    ``modeling.monitor.psi`` is a symmetrized KL over quantile bins, and classifier ``log_loss``
    is cross-entropy = H(P) + KL(P ‖ Q) — minimizing it minimizes the divergence from truth.
    """
    return float(stats.entropy(np.asarray(p, dtype=float), np.asarray(q, dtype=float), base=base))


def information_gain(df: pl.DataFrame, feature: str, target: str, *, base: float = 2.0) -> float:
    """Entropy drop in ``target`` from knowing categorical ``feature`` — the tree-split criterion.

    H(target) - Σ p(level)·H(target | level), in bits: 0 means the feature tells you nothing;
    H(target) means it determines the target. Equals the mutual information between the two
    categorical columns (``mutual_information`` is the numeric-feature counterpart).
    """
    sub = df.select([feature, target]).drop_nulls()
    total = entropy(sub[target].cast(pl.String).to_numpy(), base=base)
    conditional = 0.0
    for level in sub.select(feature).unique().to_series().to_list():
        group = sub.filter(pl.col(feature) == level)
        share = group.height / sub.height
        conditional += share * entropy(group[target].cast(pl.String).to_numpy(), base=base)
    return total - conditional


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
