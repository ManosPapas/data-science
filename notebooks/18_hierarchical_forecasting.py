# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 18 · Hierarchical forecasting — total, regions, and countries that add up
#
# Forecast the company line, the regions, and the countries independently and the levels will
# disagree — finance gets three answers to one question. Reconciliation restores coherence, and
# the projection method (`ols`) usually *improves* accuracy while doing it, because every level's
# information gets pooled. We build a two-level geography (total → EU/NA, EU → UK/DE), forecast
# each node with ETS, measure the disagreement, reconcile three ways, and score everything
# against held-out actuals.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Four years of monthly demand, two levels of geography
# Each leaf series has its own level, trend, and seasonality (UK is the most seasonal); the
# aggregates are the exact sums. The last 12 months are held out as the test set.

# %%
months = 48
t = np.arange(months)
season = np.sin(2 * np.pi * t / 12)
leaves = {
    "UK": 300 + 2.0 * t + 60 * season + rng.normal(0, 12, months),
    "DE": 500 + 4.0 * t + 35 * season + rng.normal(0, 15, months),
    "NA": 900 + 6.0 * t + 50 * np.roll(season, 6) + rng.normal(0, 25, months),
}
series = dict(leaves)
series["EU"] = series["UK"] + series["DE"]
series["total"] = series["EU"] + series["NA"]
geography = {"total": ["EU", "NA"], "EU": ["UK", "DE"]}

horizon = 12
train_series = {name: values[:-horizon] for name, values in series.items()}
actuals = {name: values[-horizon:] for name, values in series.items()}

history = pl.DataFrame({"month": t, **series})
interactive.line(
    history.unpivot(on=list(series), index="month"),
    "month",
    "value",
    color="variable",
    title="The hierarchy: total = EU + NA, EU = UK + DE",
)

# %% [markdown]
# ## 2. Forecast every node independently — and measure the disagreement
# One ETS (trend + seasonality) per node, each fit only on its own history. Independent models
# make independent errors, so the parents stop equalling the sum of their children — the
# coherence gap below is what finance would call "the numbers don't tie".

# %%
base_forecasts = {}
for name, values in train_series.items():
    model_ets = make_forecaster("ets", trend="add", seasonal="add", seasonal_periods=12)
    base_forecasts[name] = model_ets.fit(values).predict(horizon)

hierarchy.coherence_error(base_forecasts, geography)

# %% [markdown]
# ## 3. Reconcile three ways
# - **bottom_up** — trust the leaves, sum upward (safe when leaves are strong, noisy when thin);
# - **top_down** — trust the total, split by historical leaf shares (stable aggregate, leaf
#   accuracy only as good as the split);
# - **ols** — project *all* forecasts onto the coherent subspace: every level's information used.
# After any of them the levels tie exactly; the question left is which is most *accurate*.

# %%
leaf_shares = {
    name: float(train_series[name][-12:].sum())
    / float(sum(train_series[leaf][-12:].sum() for leaf in ["UK", "DE", "NA"]))
    for name in ["UK", "DE", "NA"]
}
reconciled = {
    "base": base_forecasts,
    "bottom_up": hierarchy.reconcile(base_forecasts, geography, method="bottom_up"),
    "top_down": hierarchy.reconcile(
        base_forecasts, geography, method="top_down", proportions=leaf_shares
    ),
    "ols": hierarchy.reconcile(base_forecasts, geography, method="ols"),
}
for method in ("bottom_up", "top_down", "ols"):
    gaps = hierarchy.coherence_error(reconciled[method], geography)
    print(f"{method:10s} max coherence gap: {gaps['mean_abs_gap'].max():.6f}")

# %% [markdown]
# ## 4. Score against the held-out year
# MAE per node and method. The pattern to expect: bottom-up wins where leaves are clean, top-down
# protects the total but smears the leaves, and OLS is the consistent all-rounder — it shares
# the levels' information instead of anointing one of them.

# %%
rows = []
for method, forecasts in reconciled.items():
    for node in series:
        rows.append(
            {
                "method": method,
                "node": node,
                "mae": backtest.mae(actuals[node], forecasts[node]),
            }
        )
scores = pl.DataFrame(rows).pivot(on="method", index="node", values="mae")
scores.with_columns(pl.exclude("node").round(1))

# %%
mean_by_method = scores.select(pl.exclude("node")).mean()
print("average MAE across nodes by method:")
print(mean_by_method)

# %%
comparison = pl.DataFrame(
    {
        "month": np.arange(horizon),
        "actual": actuals["total"],
        "base": reconciled["base"]["total"],
        "ols": reconciled["ols"]["total"],
    }
)
interactive.line(
    comparison.unpivot(on=["actual", "base", "ols"], index="month"),
    "month",
    "value",
    color="variable",
    title="Total line: base vs OLS-reconciled forecast against actuals",
)

# %% [markdown]
# **Takeaways:** independently fitted nodes disagreed by a visible coherence gap before
# reconciliation and tie to machine precision after it — the "numbers that don't add up" failure
# mode is structural, not a data bug, whenever levels are forecast separately; all three methods
# restore coherence but they are *not* interchangeable on accuracy: top-down keeps the stable
# total while degrading country lines, bottom-up rides the quality of the leaves, and OLS
# reconciliation pools every level's information and lands at-or-better than the base forecasts
# on average — making it the sensible default, with the others as deliberate choices when you
# trust one level far more than the rest. Operationally: forecast every node with the usual
# tooling (notebook 05), then make `reconcile(...)` the last step before anything ships to
# planning or finance.
