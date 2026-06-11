# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 06 · Scale — lazy Polars & DuckDB over files
#
# The repo's reason for being: work on data **larger than RAM** by *scanning* rather than loading.
# Lazy Polars pushes filters and column selection down into the Parquet scan, and DuckDB runs SQL
# straight over files (out-of-core). Shown on tiny samples here; the same code holds at GBs.

# %%
from core.config import ROOT
from core.prelude import *

RAW = ROOT / "data" / "raw"

# %% [markdown]
# ## 1. Lazy Polars — scan, don't load
# `scan_parquet` returns a **LazyFrame**: nothing is read until `.collect()`. Build the whole query
# first, then execute it once.

# %%
lf = scan_parquet(RAW / "customers.parquet")
type(lf).__name__  # 'LazyFrame' — no data in memory yet

# %%
premium_by_region = (
    lf.filter(pl.col("plan") == "premium")
    .group_by("region")
    .agg(
        pl.len().alias("customers"),
        pl.col("monthly_spend").mean().round(2).alias("avg_spend"),
    )
    .sort("customers", descending=True)
)
# The optimized plan shows projection & predicate pushdown into the scan:
print(premium_by_region.explain())

# %%
premium_by_region.collect()  # only now is anything read — and only the columns/rows needed

# %% [markdown]
# ## 2. DuckDB — SQL straight over files (out-of-core)
# `query_files` runs DuckDB SQL over Parquet or CSV without a load step; it streams over the files,
# so it stays bounded even when inputs exceed RAM.

# %%
customers_path = (RAW / "customers.parquet").as_posix()
query_files(
    f"SELECT region, count(*) AS customers, round(avg(monthly_spend), 2) AS avg_spend "
    f"FROM '{customers_path}' GROUP BY region ORDER BY customers DESC"
)

# %%
# DuckDB reads CSV directly too — handy for raw exports before they're converted to Parquet.
transactions_path = (RAW / "transactions.csv").as_posix()
query_files(
    f"SELECT segment, count(*) AS orders, round(sum(revenue)) AS revenue "
    f"FROM '{transactions_path}' WHERE segment IS NOT NULL "
    f"GROUP BY segment ORDER BY revenue DESC"
)

# %% [markdown]
# ## 3. A heavier lazy aggregate
# Yearly revenue from the daily series — built lazily, collected in a single pass. For inputs that
# don't fit in memory, add `engine="streaming"` to `.collect()` to process in chunks.

# %%
(
    scan_parquet(RAW / "daily_sales.parquet")
    .with_columns(pl.col("date").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.col("revenue").sum().round(0).alias("revenue"),
        pl.col("orders").sum().alias("orders"),
    )
    .sort("year")
    .collect()
)

# %% [markdown]
# **Takeaways:** start from `scan_*` / `query_files`, select columns and filter rows early, and let
# the engine push that work into the scan — the same code scales from these samples to gigabytes.
