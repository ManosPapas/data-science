"""Time-series diagnostics — stationarity, trend, seasonality, breaks, residual autocorrelation.

Run before modelling (ADF/KPSS decide differencing or detrending; ``trend_test`` and
``dominant_period`` characterize drift and the seasonal period; ``change_points`` finds level
breaks worth investigating) and after (``ljung_box`` on residuals — leftover autocorrelation means
the model missed structure). Charts live in ``viz.timeseries`` (acf/pacf, rolling_stats,
seasonal_decomposition).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.analytics.stats import TestResult


def adf_test(series: ArrayLike) -> TestResult:
    """Augmented Dickey-Fuller. H0: unit root (non-stationary); p < 0.05 → stationary.

    Most classical forecasters (ARIMA's AR/MA parts) assume a stable mean/variance — model a
    non-stationary series directly and spurious correlations follow. Pair with :func:`kpss_test`
    (opposite null) via :func:`stationarity_report` rather than reading either alone.
    """
    from statsmodels.tsa.stattools import adfuller

    statistic, p_value, *_ = adfuller(np.asarray(series, dtype=float))
    return TestResult(float(statistic), float(p_value))


def kpss_test(series: ArrayLike, *, regression: str = "c") -> TestResult:
    """KPSS. H0: stationary around a level ('c') or trend ('ct') — the mirror image of ADF.

    p < 0.05 rejects stationarity. p-values are clipped to the [0.01, 0.1] lookup-table range
    (statsmodels interpolates a printed table), so read extremes as bounds, not exact values.
    """
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)  # p clipped to table range
        statistic, p_value, *_ = kpss(np.asarray(series, dtype=float), regression=regression)
    return TestResult(float(statistic), float(p_value))


def stationarity_report(series: ArrayLike, *, alpha: float = 0.05) -> dict[str, float | str]:
    """ADF + KPSS combined into one verdict: stationary / difference it / detrend it.

    The two tests have opposite nulls, so together they distinguish a unit root (difference the
    series) from a deterministic trend (subtract the trend) — differencing a trend-stationary
    series, or detrending a unit-root one, leaves the problem in place.
    """
    adf = adf_test(series)
    kp = kpss_test(series)
    adf_stationary = adf.p_value < alpha  # ADF rejects its unit-root null
    kpss_stationary = kp.p_value >= alpha  # KPSS fails to reject its stationary null
    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary: difference it"
    elif adf_stationary:
        verdict = "difference-stationary: difference it"
    else:
        verdict = "trend-stationary: detrend it"
    return {
        "adf_statistic": adf.statistic,
        "adf_p": adf.p_value,
        "kpss_statistic": kp.statistic,
        "kpss_p": kp.p_value,
        "verdict": verdict,
    }


def ljung_box(residuals: ArrayLike, *, lags: int = 10) -> TestResult:
    """Ljung-Box on model residuals. H0: no autocorrelation up to ``lags``.

    A good forecaster leaves white-noise residuals; a small p-value means predictable structure
    was missed (raise the AR/MA order, add the seasonal term, add features). Set ``lags`` to at
    least one seasonal cycle.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    frame = acorr_ljungbox(np.asarray(residuals, dtype=float), lags=[lags])
    return TestResult(float(frame["lb_stat"].iloc[0]), float(frame["lb_pvalue"].iloc[0]))


def trend_test(series: ArrayLike, *, alpha: float = 0.05) -> dict[str, float | str]:
    """Mann-Kendall trend test + Sen's slope — is the series drifting, and how fast?

    Non-parametric: assumes no linearity or normality and shrugs off outliers. A small p-value
    says a monotonic trend exists; ``slope`` (Sen's: the median pairwise slope) is the robust
    per-step rate of change. Strong seasonality inflates the test — deseasonalize first
    (``viz.timeseries.seasonal_decomposition``).
    """
    from scipy.stats import norm

    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 8:
        raise ValueError("trend_test needs at least 8 observations")
    s = 0.0
    pairwise_slopes: list[NDArray[np.float64]] = []
    for j in range(1, n):
        diffs = y[j] - y[:j]
        s += float(np.sign(diffs).sum())
        pairwise_slopes.append(diffs / (j - np.arange(j)))
    _, tie_counts = np.unique(y, return_counts=True)
    variance = (
        n * (n - 1) * (2 * n + 5)
        - float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    ) / 18.0
    z = (s - float(np.sign(s))) / float(np.sqrt(variance)) if variance > 0 else 0.0
    p_value = 2.0 * float(norm.sf(abs(z)))
    slope = float(np.median(np.concatenate(pairwise_slopes)))
    trend = "none" if p_value >= alpha else ("increasing" if slope > 0 else "decreasing")
    return {"statistic": z, "p_value": p_value, "slope": slope, "trend": trend}


def change_points(
    series: ArrayLike, *, min_size: int = 10, max_points: int = 5, penalty: float | None = None
) -> list[int]:
    """Mean-shift change points via binary segmentation — where did the level break?

    Recursively places the split that most reduces within-segment squared error, accepting it
    only when the gain beats ``penalty`` (default: BIC-style 2·sigma²·ln n, sigma estimated from
    first differences, so a stable series returns []). Returns sorted indices of the first point
    of each new regime — line them up with deploys, price moves, campaigns, outages.
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 2 * min_size:
        return []
    if penalty is None:
        sigma2 = float(np.diff(y).var() / 2.0) if n > 2 else float(y.var())
        penalty = 2.0 * sigma2 * float(np.log(n))
    threshold = float(penalty)

    def cost(segment: NDArray[np.float64]) -> float:
        return float(((segment - segment.mean()) ** 2).sum())

    def best_split(lo: int, hi: int) -> int | None:
        base = cost(y[lo:hi])
        best_gain, best_idx = threshold, -1
        for split in range(lo + min_size, hi - min_size + 1):
            gain = base - cost(y[lo:split]) - cost(y[split:hi])
            if gain > best_gain:
                best_gain, best_idx = gain, split
        return best_idx if best_idx >= 0 else None

    found: list[int] = []
    queue: list[tuple[int, int]] = [(0, n)]
    while queue and len(found) < max_points:
        lo, hi = queue.pop(0)
        if hi - lo < 2 * min_size:
            continue
        split = best_split(lo, hi)
        if split is None:
            continue
        found.append(split)
        queue.extend([(lo, split), (split, hi)])
    return sorted(found)


def dominant_period(series: ArrayLike) -> int:
    """Strongest cycle length via the periodogram (e.g. 7 on daily data = weekly seasonality).

    Linearly detrends first so a trend doesn't drown the seasonal peak. On weakly seasonal data
    the spectral peak can be noise — confirm with ``viz.timeseries.acf`` / ``seasonal_subseries``
    before wiring the period into a model.
    """
    from scipy.signal import periodogram

    values = np.asarray(series, dtype=float)
    if values.size < 4:
        raise ValueError("series too short to estimate a period")
    freqs, power = periodogram(values, detrend="linear")
    best = int(np.argmax(power[1:])) + 1  # skip the zero-frequency bin
    return round(1.0 / float(freqs[best]))
