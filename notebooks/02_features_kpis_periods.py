# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 · Features, KPIs & period comparisons
#
# Starting from the **cleaned** transactions (notebook 01) plus the customer master, build
# features, compute financial & behaviour KPIs, and slice/compare the business over time
# (MoM / QoQ / YoY).
#
# > Run notebook **01** first — it writes `data/processed/transactions_clean.parquet`.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
RAW = ROOT / "data" / "raw"

tx = read_parquet(ROOT / "data" / "processed" / "transactions_clean.parquet")
customers = read_parquet(RAW / "customers.parquet")
tx.head()

# %% [markdown]
# ## 1. Feature engineering
# Calendar parts, share-of-total, quantile bins, and an enriching join to the customer master.

# %%
tx = temporal.add_calendar(tx, "order_date")
tx.select(["order_date", "year", "quarter", "month", "weekday", "is_weekend"]).head()

# %%
tx = transform.pct_of_total(tx, "revenue", over="region")
tx = transform.discretize(
    tx, "revenue", quantiles=[0.25, 0.5, 0.75], labels=["low", "mid", "high", "top"]
)
tx.select(["region", "revenue", "revenue_pct", "revenue_bin"]).head()

# %%
# Join to the customer master. Align the key dtype first (the clean frame was downcast to Int32).
right = customers.select(["customer_id", "plan", "age"]).with_columns(
    pl.col("customer_id").cast(tx["customer_id"].dtype)
)
enriched = transform.join(tx, right, on="customer_id", how="left", validate="m:1")
enriched.select(["order_id", "customer_id", "plan", "age", "revenue"]).head()

# %% [markdown]
# ## 2. Financial KPIs
# Scalar KPIs from `core.kpi.financial` (each takes prepared numbers → one value).

# %%
total_revenue = float(tx["revenue"].sum())
n_orders = tx.height
n_customers = tx["customer_id"].n_unique()
assumed_cogs = total_revenue * 0.62  # illustrative cost base

financial_kpis = {
    "AOV": financial.average_order_value(total_revenue, n_orders),
    "ARPU": financial.arpu(total_revenue, n_customers),
    "GMV": financial.gmv(tx["revenue"].to_numpy()),
    "gross_margin": financial.gross_margin(total_revenue, assumed_cogs),
    "take_rate (8% fee)": financial.take_rate(total_revenue * 0.08, total_revenue),
    "ROAS (5% spend)": financial.roas(total_revenue, total_revenue * 0.05),
}
pl.DataFrame({"kpi": list(financial_kpis), "value": list(financial_kpis.values())})

# %%
# Revenue by year, with compound annual growth across the window.
yearly = transform.aggregate(tx, "year", {"revenue": "sum"}).sort("year")
cagr = financial.cagr(yearly["revenue_sum"][0], yearly["revenue_sum"][-1], yearly.height - 1)
print(f"CAGR across {yearly.height} years: {cagr:.1%}")
yearly.with_columns(pl.col("revenue_sum").pct_change().alias("yoy_growth"))

# %%
# Per-segment revenue, orders and AOV.
by_segment = transform.aggregate(tx, "segment", {"revenue": "sum", "order_id": "count"})
by_segment.with_columns(
    (pl.col("revenue_sum") / pl.col("order_id_count")).round(2).alias("aov")
).sort("revenue_sum", descending=True)

# %% [markdown]
# ## 3. Behaviour KPIs
# Repeat purchasing, a conversion funnel, and NPS from the customer satisfaction scores.

# %%
orders_per_customer = tx.group_by("customer_id").len()
repeat_customers = int((orders_per_customer["len"] > 1).sum())
repeat = behaviour.repeat_rate(repeat_customers, orders_per_customer.height)
nps_value = behaviour.nps(customers["satisfaction"].drop_nulls().to_numpy())
print(f"repeat rate: {repeat:.1%}   NPS (from satisfaction): {nps_value:.0f}")

# %%
behaviour.funnel([10000, 6200, 3100, 1850], ["visits", "carts", "checkout", "purchase"])

# %%
# Stage-by-stage rates from the same funnel numbers — where exactly do we leak?
print(f"add-to-cart  {behaviour.add_to_cart_rate(6200, 10000):.0%}")
print(f"checkout     {behaviour.checkout_rate(3100, 6200):.0%}")
print(f"abandonment  {behaviour.cart_abandonment_rate(1850, 3100):.0%} of checkouts never purchase")

# %% [markdown]
# ### Cohort retention — when do customers go quiet?
# Group customers by *first-order* month, track the share ordering again k months later, and read
# the heatmap row by row. Survivorship guard: every customer who ever ordered stays in their
# cohort's denominator forever — dropping the quiet ones would flatter every number.

# %%
first_order = tx.group_by("customer_id").agg(
    pl.col("order_date").min().dt.truncate("1mo").alias("cohort")
)
cohort_sizes = first_order.group_by("cohort").agg(pl.len().alias("size"))
activity = (
    tx.join(first_order, on="customer_id")
    .with_columns(
        (
            (pl.col("order_date").dt.year() - pl.col("cohort").dt.year()) * 12
            + (pl.col("order_date").dt.month() - pl.col("cohort").dt.month())
        ).alias("months_since")
    )
    .filter(pl.col("months_since") <= 6)
    .group_by("cohort", "months_since")
    .agg(pl.col("customer_id").n_unique().alias("active"))
)
retention = (
    activity.join(cohort_sizes, on="cohort")
    .with_columns((pl.col("active") / pl.col("size")).alias("retention"))
    .pivot("months_since", index="cohort", values="retention")
    .sort("cohort")
    .head(9)
    .with_columns(cs.numeric().fill_null(0.0))  # a missing cell means nobody ordered: 0%
    .with_columns(pl.col("cohort").dt.strftime("%Y-%m"))
)
fig, axes = base.grid(1, ncols=1)
timeseries.cohort_heatmap(retention, index="cohort", ax=axes[0], title="Repeat-order retention")

# %% [markdown]
# ## 4. Time-period slicing & comparison
# Default rolling windows, then period-over-period comparisons (the bread and butter of reporting).

# %%
recent = period.window(tx, "order_date", "90d")
print(f"last 90 days: {recent.height} orders, {recent['revenue'].sum():,.0f} revenue")

# %%
mom = period.month_over_month(tx, "order_date", "revenue")
qoq = period.quarter_over_quarter(tx, "order_date", "revenue")
yoy = period.year_over_year(tx, "order_date", "revenue")
pl.DataFrame(
    {
        "comparison": ["MoM", "QoQ", "YoY"],
        "current": [mom.current, qoq.current, yoy.current],
        "previous": [mom.previous, qoq.previous, yoy.previous],
        "pct_change": [mom.pct_change, qoq.pct_change, yoy.pct_change],
    }
)

# %%
# Quarter-over-quarter revenue per region in one shot.
period.compare_periods_by(tx, "order_date", "revenue", "region", period="quarter")

# %%
# Any two explicit windows compare the same way — e.g. this Q2 vs Q1 (campaign vs baseline).
from datetime import date

period.compare_windows(
    tx,
    "order_date",
    "revenue",
    (date(2024, 4, 1), date(2024, 6, 30)),
    (date(2024, 1, 1), date(2024, 3, 31)),
    labels=("Q2-24", "Q1-24"),
)

# %% [markdown]
# ## 5. Group analysis
# Is revenue genuinely different across segments? Summarize, then let `compare_groups` pick the
# right test and report an effect size.

# %%
stats.group_summary(tx, "revenue", "segment")

# %%
stats.compare_groups(tx, "revenue", "segment")

# %%
segment_revenue = [
    tx.filter(pl.col("segment") == s)["revenue"].to_numpy()
    for s in ["retail", "sme", "corporate", "wealth"]
]
print("one-way ANOVA across segments:", stats.anova(*segment_revenue))

# %%
fig, axes = base.grid(2)
eda.count_bar(tx, "month", ax=axes[0], title="Orders by month")
eda.boxplot_by(tx, "revenue", "quarter", ax=axes[1], title="Revenue by quarter")

# %% [markdown]
# **Takeaways:** calendar + share + bin features and a clean customer join; a compact financial &
# behaviour KPI set with the funnel's leak points; cohort retention that keeps quiet customers in
# the denominator (no survivorship flattery); and MoM/QoQ/YoY, explicit-window, and per-region QoQ
# comparisons for reporting. Group tests quantify whether segment differences are real.
