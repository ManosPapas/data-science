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
# ## 4. The typed source catalog — register once, load by name everywhere
# A `Source` pins where a dataset lives *and* the dtypes it must satisfy; `catalog.load` enforces
# the pin and fails fast when a column disappears or stops casting. One typed loader per source,
# operationalized.

# %%
catalog.register(
    "customers",
    RAW / "customers.parquet",
    schema={"customer_id": pl.Int64, "monthly_spend": pl.Float64, "churned": pl.Int8},
)
print("registered sources:", catalog.sources())
catalog.load("customers", columns=["customer_id", "monthly_spend", "churned"]).head(3)

# %% [markdown]
# ## 5. `@cached` — expensive pulls hit Parquet the second time
# Wrap any loader that returns a DataFrame; results are cached to `data/interim/` keyed by
# function + arguments, and `refresh=True` forces a recompute. The pattern for slow warehouse
# queries: pay once per parameter set, not once per notebook run.


# %%
@cached
def premium_spend_by_region(plan: str) -> pl.DataFrame:
    return (
        scan_parquet(RAW / "customers.parquet")
        .filter(pl.col("plan") == plan)
        .group_by("region")
        .agg(pl.col("monthly_spend").mean().round(2).alias("avg_spend"))
        .sort("region")
        .collect()
    )


premium_spend_by_region("premium")  # computes and writes the cache
premium_spend_by_region("premium")  # served straight from Parquet
print([p.name for p in (ROOT / "data" / "interim").glob("premium_spend_by_region-*.parquet")])

# %% [markdown]
# ## 6. Out-of-core learning — `partial_fit` over chunks
# When even the model can't see all rows at once, stream them: slice the lazy scan, standardize
# with a scaler fitted on the first chunk, and incrementally update an SGD classifier. The same
# loop handles files far beyond RAM.

# %%
numeric_cols = ["age", "tenure_months", "num_products", "sessions_30d", "support_tickets"]
stream = scan_parquet(RAW / "customers.parquet").select([*numeric_cols, "churned"])
# average=True (Polyak averaging) is the standard stabilizer for streaming SGD — without it the
# last noisy gradient steps dominate and one-pass performance hovers near chance.
sgd = registry.make_model(
    "sgd", task="classification", loss="log_loss", average=True, alpha=1e-3, random_state=42
)
scaler = preprocess.make_scaler("standard")

chunk_size, n_rows, epochs = 1000, 5000, 3  # SGD wants a few passes over the stream
for epoch in range(epochs):
    for start in range(0, n_rows, chunk_size):
        chunk = stream.slice(start, chunk_size).collect()
        x_chunk = chunk.select(numeric_cols).to_numpy()
        first = epoch == 0 and start == 0
        scaled = scaler.fit_transform(x_chunk) if first else scaler.transform(x_chunk)
        train.partial_fit(sgd, scaled, chunk["churned"].to_numpy(), classes=np.array([0, 1]))
print(f"streamed {n_rows} rows x {epochs} epochs in {n_rows // chunk_size}-chunk passes")

# %%
# Did streaming cost us anything? Score both the streamed SGD and a full-batch logistic that was
# allowed to see everything at once — the streamed model should land in the same place.
full = stream.collect()
x_full = scaler.transform(full.select(numeric_cols).to_numpy())
y_full = full["churned"].to_numpy()
streamed_auc = evaluate.classification_metrics(
    y_full, sgd.predict(x_full), y_score=sgd.predict_proba(x_full)[:, 1]
)["roc_auc"]
batch = train.fit(
    registry.make_model("logistic", task="classification", max_iter=1000), x_full, y_full
)
batch_auc = evaluate.classification_metrics(
    y_full, batch.predict(x_full), y_score=batch.predict_proba(x_full)[:, 1]
)["roc_auc"]
print(f"streamed SGD ROC-AUC {streamed_auc:.3f}  vs  full-batch logistic {batch_auc:.3f}")

# %% [markdown]
# **Takeaways:** start from `scan_*` / `query_files`, select columns and filter rows early, and let
# the engine push that work into the scan; pin schemas through the catalog, cache expensive pulls
# to Parquet, and when the model itself can't hold the data, stream it with `partial_fit` — the
# same code scales from these samples to gigabytes.
