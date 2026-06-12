# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 11 · Time-series diagnostics — interrogate the series before (and after) you model it
#
# The pre-flight checks that decide *how* to forecast daily revenue: stationarity (difference or
# detrend?), trend (how fast are we growing?), seasonality (what period?), structural breaks
# (did the level shift — and when?), and residual whiteness (did the model capture it all?).
# Complements notebook 05, which compares the forecasters themselves.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

daily = read_parquet(ROOT / "data" / "raw" / "daily_sales.parquet").sort("date")
y = daily["revenue"].to_numpy()

# %% [markdown]
# ## 1. Stationarity — one verdict from two opposite tests
# ADF (H0: unit root) and KPSS (H0: stationary) read together: the raw series is non-stationary;
# its first difference is stationary — so ARIMA-style models want d=1, exactly what the `(2,1,2)`
# order in notebook 05 encodes.

# %%
print("raw series :", diagnostics.stationarity_report(y))
print("differenced:", diagnostics.stationarity_report(np.diff(y)))

# %% [markdown]
# ## 2. Trend — direction and an honest rate
# Mann-Kendall + Sen's slope is non-parametric (no linearity/normality assumed, outlier-proof).
# Weekly seasonality inflates the daily test, so we also run it on weekly means — the slope there
# reads directly as revenue growth per week (the generator's truth is +0.45/day ≈ +3.2/week).

# %%
weekly = transform.resample(
    daily, time_col="date", every="1w", aggs=[pl.col("revenue").mean().alias("revenue")]
)
print("daily :", diagnostics.trend_test(y))
print("weekly:", diagnostics.trend_test(weekly["revenue"].to_numpy()))

# %% [markdown]
# ## 3. Seasonality — let the periodogram name the cycle
# On the full six years the strongest cycle is the *yearly* swing (its amplitude beats the weekly
# pattern); zoom into a recent quarter — where the yearly cycle reads as trend and is detrended
# away — and the weekly cycle wins. That 7 is the `seasonal_periods`/`season_length` every
# forecaster in 05 uses; the ACF and the weekly profile confirm it.

# %%
print(f"dominant cycle, full history : {diagnostics.dominant_period(y)} days")
print(f"dominant cycle, last 120 days: {diagnostics.dominant_period(y[-120:])} days")
fig, axes = base.grid(2)
timeseries.acf(y, lags=30, ax=axes[0], title="ACF — spikes at 7, 14, 21")
timeseries.seasonal_subseries(y, period=7, ax=axes[1], title="Weekly profile")

# %% [markdown]
# ## 4. Change points — did the level break?
# The detector hunts *sustained mean shifts*, so give it a level-style series. First a clean case:
# a daily conversion-rate with a silent regression after a deploy on day 140 — binary segmentation
# pins the break. Then the revenue series: a trend looks like a staircase of mean shifts to any
# break detector, so detrend first — and read what remains carefully: the flags land on *seasonal*
# plateaus (the yearly cycle's swings, and the early-December promo step), because the detector
# reports sustained shifts of any origin. Deseasonalize too when you only want one-off breaks.

# %%
deploy_day = 140
conversion = np.where(np.arange(240) < deploy_day, 0.062, 0.055) + rng.normal(0.0, 0.002, 240)
found = diagnostics.change_points(conversion, min_size=20)
print(f"injected break at day {deploy_day}; detected at {found}")

# %%
trend_line = np.polyval(np.polyfit(np.arange(y.size), y, 1), np.arange(y.size))
flagged = diagnostics.change_points(y - trend_line, min_size=30)
print("level shifts in detrended revenue:", [(i, daily["date"][int(i)]) for i in flagged])

# %% [markdown]
# ## 5. Residual whiteness — is there signal left on the table?
# A naive forecaster's residuals (= first differences) still carry the weekly cycle — Ljung-Box
# rejects loudly. A seasonal ETS fitted on train and scored on a 28-day holdout leaves residuals
# the test can't tell from noise: the model has captured the structure.

# %%
print("naive residuals  :", diagnostics.ljung_box(np.diff(y), lags=14))

horizon = 28
ets = make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=7)
ets.fit(y[:-horizon])
residuals = y[-horizon:] - ets.predict(horizon)
print("ETS holdout resid:", diagnostics.ljung_box(residuals, lags=7))

# %%
fig, axes = base.grid(1, ncols=1)
timeseries.forecast_residuals(
    residuals,
    dates=daily["date"].to_numpy()[-horizon:],
    ax=axes[0],
    title="ETS 28-day holdout residuals",
)

# %% [markdown]
# **Takeaways:** the series is difference-stationary with a ~+3/week trend and a 7-day cycle —
# which fixes the model family (d=1, weekly seasonal term) before any fitting; the break detector
# pinpoints injected level shifts to the day (run it on differenced/detrended data); and Ljung-Box
# certifies the seasonal ETS leaves nothing forecastable behind, while the naive model does.
