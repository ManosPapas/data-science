# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Pax-revenue · Phase 4 — Demand forecasting (the volume side)
#
# **Decision this informs:** how many seats will sell on these five long-hauls, with honest
# uncertainty — the volume input to the oil/FX margin model in Phase 6, and the early-warning on
# routes pacing soft.
#
# Method (playbook §3): diagnose the series → **always** beat naive baselines → ETS / SARIMAX / ML
# → a **rolling-origin backtest** picks the winner (never in-sample fit) → **prediction intervals**
# (a point forecast is not a deliverable). The per-route bake-off is **parallelised** across the ten
# directional routes.
#
# **Sampling note:** we keep *every* booking on the five chosen pairs (complete daily series), not a
# random row sample — random sampling would shred the time series. That is the statistically correct
# way to make this fast.
#
# > Input: `data/processed/pax_revenue_enriched.parquet`

# %%
import time
from datetime import timedelta

from joblib import Parallel, delayed

from core.config import ROOT
from core.prelude import *

set_theme()
ENRICHED = ROOT / "data" / "processed" / "pax_revenue_enriched.parquet"

fares = read_parquet(ENRICHED).filter(~pl.col("is_refund"))
route_codes = fares["route_code"].unique().to_list()
rc_to_route = dict(fares.select("route_code", "route").unique().iter_rows())
print(f"{fares.height:,} paid fares across {len(route_codes)} routes")

# %% [markdown]
# ## 1. Build the daily demand series (by departure date)
# Target = pax-segments per **departure date** (scheduled/flown demand). We forecast by departure
# date, **not** booking date: booking-intake-by-booking-date is *right-censored* at the extract
# (bookings for future travel are still arriving), so its tail droops and any model just extrapolates
# the droop. Departure-date demand is clean on the right; the *left* edge is mildly undercounted
# (early-2025 departures had some bookings made before the data window opened), which inflates the
# apparent early-period trend — flagged, not modelled away. Zero-fill on a shared range so routes align.

# %%
gmin = fares["departure_time"].dt.date().min()
gmax = fares["departure_time"].dt.date().max()
calendar = pl.DataFrame({"date": pl.date_range(gmin, gmax, "1d", eager=True)})


def daily_series(frame: pl.DataFrame) -> pl.DataFrame:
    """Daily pax count by departure date on the shared calendar (missing days -> 0)."""
    counts = (
        frame.group_by(pl.col("departure_time").dt.date().alias("date"))
        .agg(pl.len().alias("pax"))
    )
    return calendar.join(counts, on="date", how="left").with_columns(
        pl.col("pax").fill_null(0)
    ).sort("date")


headline = daily_series(fares)
y = headline["pax"].to_numpy().astype(float)
print(f"daily series: {y.size} days, mean {y.mean():.0f} pax/day, min {y.min():.0f}, max {y.max():.0f}")
headline.tail()

# %% [markdown]
# ## 2. Diagnose the series first
# Stationarity (ADF+KPSS verdict), the dominant cycle, the trend, and any level breaks — these
# decide which models can even be trusted.

# %%
print("stationarity:", diagnostics.stationarity_report(y))
print("dominant period (days):", diagnostics.dominant_period(y))
print("trend:", diagnostics.trend_test(y))
print("change-point indices:", diagnostics.change_points(y, min_size=21))

# %%
fig, axes = base.grid(3, ncols=3)
timeseries.rolling_stats(y, window=14, ax=axes[0], title="Rolling mean & std (14d)")
timeseries.acf(y, lags=35, ax=axes[1], title="ACF")
timeseries.pacf(y, lags=35, ax=axes[2], title="PACF")

# %%
# Weekly (period 7) trend/seasonal/residual split. Note the dominant periodogram cycle is *longer*
# than a week (annual leisure seasonality) — but one year of data can identify the weekly period,
# not a full annual one, so 7 is what we can model. The trend is partly the left-edge artifact above.
timeseries.seasonal_decomposition(y, period=7)

# %% [markdown]
# ## 3. Model bake-off on the headline series (rolling-origin backtest)
# Every model is scored the same way: expanding-window train, forecast 14 days, roll forward. The
# baselines are the bar to clear — a model that can't beat `seasonal_naive` isn't earning its keep.

# %%
HEADLINE_MODELS = {
    "naive": lambda: make_forecaster("naive"),
    "mean": lambda: make_forecaster("mean"),
    "seasonal_naive": lambda: make_forecaster("seasonal_naive", season_length=7),
    "ets": lambda: make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=7),
    "sarimax": lambda: make_forecaster("sarimax", order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)),
    "ml_rf": lambda: make_forecaster(
        "ml",
        estimator=registry.make_model("random_forest", task="regression",
                                       n_estimators=200, n_jobs=-1, random_state=42),
        lags=14,
    ),
}


def bake_off(series: np.ndarray, models: dict, *, initial: int, horizon: int = 14, step: int = 14) -> pl.DataFrame:
    """Rolling-origin MAE/RMSE/MAPE per model, best (lowest MAE) first."""
    rows = []
    for name, make in models.items():
        try:
            bt = backtest.rolling_origin(make, series, initial=initial, horizon=horizon, step=step)
            a, p = bt["actual"].to_numpy(), bt["pred"].to_numpy()
            rows.append({"model": name, "mae": round(backtest.mae(a, p), 2),
                         "rmse": round(backtest.rmse(a, p), 2), "mape": round(backtest.mape(a, p), 1)})
        except Exception as exc:  # noqa: BLE001 — a model that won't fit is just disqualified
            rows.append({"model": name, "mae": float("inf"), "rmse": float("inf"),
                         "mape": float("inf"), "error": str(exc)[:40]})
    return pl.DataFrame(rows).sort("mae")


board = bake_off(y, HEADLINE_MODELS, initial=int(y.size * 0.8))
board

# %%
winner = board["model"][0]
naive_mae = board.filter(pl.col("model") == "seasonal_naive")["mae"][0]
best_mae = board["mae"][0]
print(f"winner: {winner} | MAE {best_mae:.1f} vs seasonal_naive {naive_mae:.1f} "
      f"-> {1 - best_mae / naive_mae:+.1%} skill over naive")

# %% [markdown]
# ## 4. Forecast the next 28 days, with a prediction interval
# Refit the winner on the full series and forecast ahead. The band widening with horizon is the
# honest compounding of uncertainty.

# %%
H = 28
f = HEADLINE_MODELS[winner]()
f.fit(y)
point = np.clip(f.predict(H), 0, None)
lo, hi = f.predict_interval(H, alpha=0.05, point=point)
lo, hi = np.clip(lo, 0, None), np.clip(hi, 0, None)

tail = 120
future = pl.date_range(gmax + timedelta(days=1), gmax + timedelta(days=H), "1d", eager=True).to_list()
dates = headline["date"].to_list()[-tail:] + future
actual = list(y[-tail:]) + [np.nan] * H
predicted = [np.nan] * tail + list(point)
lower = [np.nan] * tail + list(lo)
upper = [np.nan] * tail + list(hi)

fig, axes = base.grid(1, ncols=1, figsize=(12, 5))
timeseries.forecast(dates, actual, predicted, lower=lower, upper=upper, ax=axes[0],
                    title=f"Total daily departures (pax) — {winner} forecast, next {H} days (95% PI)")
print(f"next {H}d total demand: {point.sum():,.0f} pax  [{lo.sum():,.0f}, {hi.sum():,.0f}]")

# %% [markdown]
# ## 5. Per-route forecasts — parallelised across the ten routes
# The same expanding-window backtest, run concurrently (`joblib` threads). Each route picks its own
# winner among a fast model set; the RF uses `n_jobs=1` here because the *routes* are already
# parallel (no nested over-subscription). Output: best model, backtest MAE, skill over naive, and a
# 28-day demand forecast with interval — per route.

# %%
ROUTE_MODELS = {
    "seasonal_naive": lambda: make_forecaster("seasonal_naive", season_length=7),
    "ets": lambda: make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=7),
    "ml_rf": lambda: make_forecaster(
        "ml",
        estimator=registry.make_model("random_forest", task="regression",
                                       n_estimators=100, n_jobs=1, random_state=42),
        lags=14,
    ),
}


def evaluate_route(route_code: str, h: int = 28) -> dict:
    series = daily_series(fares.filter(pl.col("route_code") == route_code))["pax"].to_numpy().astype(float)
    initial = max(60, int(series.size * 0.8))
    scores = {}
    for name, make in ROUTE_MODELS.items():
        try:
            bt = backtest.rolling_origin(make, series, initial=initial, horizon=14, step=14)
            scores[name] = backtest.mae(bt["actual"].to_numpy(), bt["pred"].to_numpy())
        except Exception:  # noqa: BLE001
            scores[name] = float("inf")
    best = min(scores, key=lambda k: scores[k])
    naive = scores.get("seasonal_naive", float("nan"))
    fc = ROUTE_MODELS[best]()
    fc.fit(series)
    pt = np.clip(fc.predict(h), 0, None)
    plo, phi = fc.predict_interval(h, alpha=0.05, point=pt)
    return {
        "route": rc_to_route[route_code],
        "best_model": best,
        "mae": round(scores[best], 2),
        "naive_mae": round(naive, 2),
        "skill": round(1 - scores[best] / naive, 3) if naive and np.isfinite(naive) and naive > 0 else None,
        "fc_28d": round(float(pt.sum()), 0),
        "lo_28d": round(float(np.clip(plo, 0, None).sum()), 0),
        "hi_28d": round(float(np.clip(phi, 0, None).sum()), 0),
    }


t0 = time.perf_counter()
results = Parallel(n_jobs=-1, prefer="threads")(delayed(evaluate_route)(rc) for rc in route_codes)
route_fc = pl.DataFrame(results).sort("fc_28d", descending=True)
print(f"forecast {len(route_codes)} routes in {time.perf_counter() - t0:.1f}s (parallel)")
route_fc

# %%
# Next-28-day demand per route with the 95% interval as error bars.
rf = route_fc.sort("fc_28d", descending=True)
fig, axes = base.grid(1, ncols=1, figsize=(11, 5))
yerr = [(rf["fc_28d"] - rf["lo_28d"]).to_numpy(), (rf["hi_28d"] - rf["fc_28d"]).to_numpy()]
axes[0].bar(rf["route"], rf["fc_28d"], yerr=yerr, capsize=4, color="tab:blue", alpha=0.8)
axes[0].set(title="Next-28-day demand forecast by route (95% PI)", ylabel="pax")
axes[0].tick_params(axis="x", rotation=45)

# %% [markdown]
# **Takeaways**
#
# - **Target choice is the senior call here:** forecasting by *departure date* (flown demand), not
#   booking date — booking-intake is right-censored at the extract and collapses any forecast. This
#   one decision is what makes the numbers usable.
# - **Disciplined backtest, honest verdict:** here the `seasonal_naive` weekly-repeat *wins* —
#   ETS/SARIMAX/ML don't beat it on one year of strongly-seasonal demand. The value is *proving*
#   that with a rolling-origin backtest, not shipping an over-engineered model. Forecasts still ship
#   **with intervals**; the per-route table gives each route's best model, backtest MAE, skill over
#   naive, and a 28-day demand range — the volume input Phase 6 prices against.
# - **Parallelised** across routes (joblib threads) + `n_jobs=-1` in the headline RF — the repo's
#   sanctioned speed levers; wall-clock printed in §5.
# - **Caveats:** one year of history identifies the **weekly** cycle but not a full **annual** one,
#   so leisure long-haul forecasts beyond ~a month are directional; the apparent upward trend is
#   partly a **left-edge artifact** (early-2025 departures undercounted); sparse routes (Phuket
#   pairs) have noisier backtests — read their intervals, not the point.
#
# **Next (Phase 5):** the price side — conditional fare response / RM fare ladder per route, with
# the booking-curve confounding handled, to find where fare has room to move.
