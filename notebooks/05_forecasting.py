# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 05 · Forecasting — daily revenue
#
# Inspect a daily revenue series (trend + weekly/yearly seasonality), engineer time features,
# compare forecasters on a 28-day holdout, then evaluate properly with a rolling-origin backtest.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()

daily = read_parquet(ROOT / "data" / "raw" / "daily_sales.parquet").sort("date")
y = daily["revenue"].to_numpy()
dates = daily["date"].to_numpy()
daily.head()

# %% [markdown]
# ## 1. Inspect the series
# Decomposition isolates trend/seasonal/residual; ACF/PACF and the weekly sub-series confirm the
# 7-day cycle.

# %%
timeseries.seasonal_decomposition(y, period=7)

# %%
# Stationarity verdict justifies d=1 in the ARIMA order below; notebook 11 runs the full
# diagnostic suite (trend, dominant period, change points, residual whiteness).
print(diagnostics.stationarity_report(y)["verdict"])

# %%
fig, axes = base.grid(4, ncols=2)
timeseries.rolling_stats(y, window=30, ax=axes[0], title="Rolling mean & std (30d)")
timeseries.acf(y, lags=40, ax=axes[1], title="ACF")
timeseries.pacf(y, lags=40, ax=axes[2], title="PACF")
timeseries.seasonal_subseries(y, period=7, ax=axes[3], title="Weekly profile")

# %% [markdown]
# ## 2. Time features
# Calendar parts, US holidays, lags, rolling means and a cyclical month encoding — the inputs you'd
# feed a feature-based forecaster (the `ml` reduction below builds its own lags internally).

# %%
featured = (
    daily.pipe(temporal.add_calendar, "date")
    .pipe(temporal.add_holiday_flags, "date", countries=["US"])
    .pipe(temporal.add_lags, "revenue", lags=[1, 7, 14])
    .pipe(temporal.add_rolling, "revenue", windows=[7, 28])
    .pipe(temporal.cyclical_encode, "month", period=12)
)
featured.select(["date", "revenue", "revenue_lag7", "revenue_roll28_mean", "is_holiday_US"]).tail()

# %% [markdown]
# ## 3. Holdout comparison (last 28 days)
# One uniform `fit(y)` / `predict(h)` interface across baselines, classical, and ML reduction.

# %%
horizon = 28
train_y, test_y = y[:-horizon], y[-horizon:]
test_dates = dates[-horizon:]

forecasters = {
    "naive": make_forecaster("naive"),
    "seasonal_naive": make_forecaster("seasonal_naive", season_length=7),
    "mean": make_forecaster("mean"),
    "ets": make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=7),
    "arima": make_forecaster("arima", order=(2, 1, 2)),
    "ml_rf": make_forecaster(
        "ml",
        estimator=registry.make_model(
            "random_forest", task="regression", n_estimators=200, random_state=42
        ),
        lags=14,
    ),
    "ml_lgbm": make_forecaster(
        "ml",
        estimator=registry.make_model(
            "lightgbm", task="regression", n_estimators=300, random_state=42, verbose=-1
        ),
        lags=14,
    ),
}
preds = {}
for name, fc in forecasters.items():
    fc.fit(train_y)
    preds[name] = fc.predict(horizon)

# %%
scores = pl.DataFrame(
    {
        "model": list(preds),
        "mae": [backtest.mae(test_y, p) for p in preds.values()],
        "rmse": [backtest.rmse(test_y, p) for p in preds.values()],
        "mape": [backtest.mape(test_y, p) for p in preds.values()],
        "smape": [backtest.smape(test_y, p) for p in preds.values()],
    }
).sort("rmse")
scores

# %%
best = scores["model"][0]
lower, upper = forecasters[best].predict_interval(horizon, point=preds[best])  # 95% interval
fig, axes = base.grid(1, ncols=1)
timeseries.forecast(
    test_dates,
    test_y,
    preds[best],
    lower=lower,
    upper=upper,
    ax=axes[0],
    title=f"28-day holdout — {best} (95% interval)",
)

# %%
# Score the band, not just the point: pinball loss is the proper loss for a quantile, so the
# 2.5%/97.5% bounds are graded as quantile forecasts (coverage should sit near 95%).
coverage = float(np.mean((test_y >= lower) & (test_y <= upper)))
lower_pinball = evaluate.pinball_loss(test_y, lower, alpha=0.025)
upper_pinball = evaluate.pinball_loss(test_y, upper, alpha=0.975)
print(
    f"interval coverage {coverage:.0%}   "
    f"pinball: lower {lower_pinball:.1f}, upper {upper_pinball:.1f}"
)

# %% [markdown]
# ## 4. Rolling-origin backtest
# A single holdout is one draw; an expanding-window backtest forecasts repeatedly and is the honest
# estimate of generalization.

# %%
bt = backtest.rolling_origin(
    lambda: make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=7),
    y,
    initial=len(y) - 180,
    horizon=14,
    step=30,
)
rmse_bt = backtest.rmse(bt["actual"], bt["pred"])
mape_bt = backtest.mape(bt["actual"], bt["pred"])
folds = bt["origin"].n_unique()
print(f"ETS rolling-origin over {folds} folds — RMSE {rmse_bt:.0f}, MAPE {mape_bt:.2f}%")

# %%
# Error typically grows with the forecast horizon h.
bt.group_by("h").agg((pl.col("actual") - pl.col("pred")).abs().mean().round(1).alias("mae")).sort(
    "h"
)

# %%
resid = (bt["actual"] - bt["pred"]).to_numpy()
fig, axes = base.grid(1, ncols=1)
timeseries.forecast_residuals(resid, ax=axes[0], title="ETS backtest residuals")

# %% [markdown]
# **Takeaways:** clear weekly seasonality and trend; ETS / seasonal-naive beat the flat baselines on
# the holdout; and the rolling-origin backtest shows error rising with horizon — the number to quote
# when committing to a forecast.
