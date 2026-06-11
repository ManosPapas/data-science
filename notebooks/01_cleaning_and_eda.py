# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 · Cleaning & EDA — transactions
#
# Take the **messy** raw transactions export and turn it into a trustworthy, memory-lean frame:
# profile it, fix dtypes, fill gaps, dedupe, tame outliers, validate, then explore distributions.
# Every non-trivial step delegates to a tested function in `core`.
#
# > Data: run `python scripts/make_sample_data.py` once to (re)generate `data/raw/`.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
RAW = ROOT / "data" / "raw"

# %% [markdown]
# ## 1. Load the raw export
# Read with `try_parse_dates=False` so we *see* the raw strings and fix types deliberately.

# %%
raw = read_csv(RAW / "transactions.csv", try_parse_dates=False)
raw.head()

# %%
print(memory_report(raw))
raw.schema

# %% [markdown]
# ## 2. Profile — what's actually in here?
# `summary` (dtype/nulls/stats per column), `missingness`, and `cardinality` (IDs vs categories).

# %%
stats.summary(raw)

# %%
stats.missingness(raw)

# %%
stats.cardinality(raw)

# %%
fig, axes = base.grid(2)
eda.missingness_bar(raw, ax=axes[0], title="Missingness (%)")
eda.count_bar(raw, "channel", ax=axes[1], title="Orders by channel")

# %% [markdown]
# ## 3. Clean
# Standardize names → normalize the messy `segment` text → fill gaps → fix dtypes →
# drop duplicate orders → winsorize outliers → downcast for memory.

# %%
# Text: " Retail", "RETAIL", "retail " all collapse to "retail"; fill the missing ones.
clean_tx = (
    raw.pipe(clean.standardize_columns)
    .pipe(clean.clean_text, ["segment"], lower=True)
    .pipe(clean.fill_missing, strategy="constant", value="unknown", columns=["segment"])
    .pipe(clean.fill_missing, strategy="median", columns=["unit_price", "discount"])
)
clean_tx["segment"].value_counts(sort=True)

# %%
# Dtypes: string date -> Date, low-cardinality strings -> Categorical (inferred from content).
print("before:", raw.schema["order_date"], "|", raw.schema["segment"])
typed = clean.auto_cast(clean_tx)
print("after: ", typed.schema["order_date"], "|", typed.schema["segment"])
typed.schema

# %%
# Duplicate order rows crept in during export — drop them on the business key.
deduped = clean.drop_duplicate_rows(typed, subset=["order_id"])
print(f"{typed.height} rows -> {deduped.height} after de-duplicating on order_id")

# %%
# Outliers: a few revenue values are wildly inflated. Quantify, then winsorize.
lo, hi = stats.outlier_bounds(deduped["revenue"].to_numpy(), method="iqr")
above = int((deduped["revenue"] > hi).sum())
print(f"IQR fence for revenue: [{lo:,.0f}, {hi:,.0f}] — {above} orders above the upper fence")
winsorized = clean.winsorize(deduped, ["revenue", "unit_price"])

# %%
print("revenue BEFORE winsorizing:", stats.describe_distribution(deduped["revenue"].to_numpy()))
print("revenue AFTER  winsorizing:", stats.describe_distribution(winsorized["revenue"].to_numpy()))

# %%
# Downcast numeric dtypes — same data, a fraction of the memory.
print(memory_report(winsorized))
compact = clean.downcast(winsorized)
print("\n" + memory_report(compact))

# %% [markdown]
# ## 4. Validate — fail fast if the contract is broken

# %%
problems = validate.check_schema(
    compact,
    required=["order_id", "order_date", "revenue"],
    non_null=["order_id", "revenue"],
    unique=["order_id"],
    ranges={"discount": (0.0, 1.0)},
    raise_on_error=False,
)
print("validation problems:", problems or "none")

# %%
# Persist the clean frame so the downstream notebooks reuse it (Parquet, not CSV).
clean_path = ROOT / "data" / "processed" / "transactions_clean.parquet"
write_parquet(compact, clean_path)
print(f"saved -> {clean_path}")

# %% [markdown]
# ## 5. Explore — distributions
# Revenue is heavily right-skewed (typical for spend); a log transform pulls it toward normal.

# %%
fig, axes = base.grid(3, ncols=3)
eda.histogram(compact["revenue"].to_numpy(), ax=axes[0], title="Revenue")
eda.histogram(compact["unit_price"].to_numpy(), ax=axes[1], title="Unit price")
eda.ecdf(compact["revenue"].to_numpy(), ax=axes[2], title="Revenue ECDF")

# %%
log_revenue = np.log1p(compact["revenue"].to_numpy())
fig, axes = base.grid(2)
eda.qq(log_revenue, ax=axes[0], title="QQ — log(1+revenue)")
eda.histogram(log_revenue, ax=axes[1], title="log(1+revenue)")
print("normality (D'Agostino):", stats.normality_test(log_revenue, method="dagostino"))

# %% [markdown]
# ## 6. Explore — relationships across categories

# %%
fig, axes = base.grid(4, ncols=2)
eda.count_bar(compact, "segment", ax=axes[0], title="Orders by segment")
eda.boxplot_by(compact, "revenue", "segment", ax=axes[1], title="Revenue by segment")
eda.crosstab_heatmap(compact, "region", "segment", ax=axes[2], title="Region x Segment")
eda.correlation_heatmap(compact, ax=axes[3], title="Numeric correlations")

# %%
sample_df = transform.sample(compact, n=800, seed=42)
eda.pairplot(sample_df, ["revenue", "unit_price", "units", "discount"], hue="segment")

# %% [markdown]
# ## 7. The same EDA, interactively (Plotly)
# `viz.interactive` mirrors the static charts with hover / zoom / pan (needs the `interactive`
# extra — `pip install plotly`).

# %%
interactive.histogram(compact, "revenue", title="Revenue (interactive)")

# %%
interactive.scatter(compact, "unit_price", "revenue", color="segment", title="Price vs revenue")

# %%
interactive.correlation_heatmap(compact, title="Numeric correlations")

# %% [markdown]
# **Takeaways:** types fixed and memory cut via downcast; `segment` cleaned + imputed; duplicate
# orders and revenue outliers handled; revenue is log-normal-ish and varies by segment. The clean
# frame is cached to `data/processed/` for the feature/KPI notebook.
