"""Time-series diagnostics — stationarity tests, residual autocorrelation, seasonality detection.

Run before modelling (ADF/KPSS decide differencing or detrending; ``dominant_period`` suggests the
seasonal period for SARIMAX/ETS) and after (``ljung_box`` on residuals — leftover autocorrelation
means the model missed structure). Charts live in ``viz.timeseries`` (acf/pacf, rolling_stats,
seasonal_decomposition).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

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
