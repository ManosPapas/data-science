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

# %%
# Is `unit_price` MCAR or MAR? Large p-values = MCAR, so a median fill won't bias; small ones
# would need conditional imputation (notebook 07, `preprocess.make_imputer`).
stats.missingness_dependence(
    raw.select(["unit_price", "units", "discount", "revenue", "segment", "region", "channel"]),
    "unit_price",
)

# %% [markdown]
# ## 3. Clean
# Standardize names → normalize the messy `segment` text → flag what was missing (the gap itself
# can be signal) → fill gaps → fix dtypes → drop duplicate orders → winsorize outliers → downcast
# for memory.

# %%
# " Retail", "RETAIL", "retail " all collapse to "retail"; flag gaps, then fill.
clean_tx = (
    raw.pipe(clean.standardize_columns)
    .pipe(clean.clean_text, ["segment"], lower=True)
    .pipe(clean.add_missing_indicators, ["unit_price", "discount"])
    .pipe(clean.fill_missing, strategy="constant", value="unknown", columns=["segment"])
    .pipe(clean.fill_missing, strategy="median", columns=["unit_price", "discount"])
)
clean_tx["segment"].value_counts(sort=True)

# %%
# string date -> Date, low-cardinality strings -> Categorical (inferred from content).
print("before:", raw.schema["order_date"], "|", raw.schema["segment"])
typed = clean.auto_cast(clean_tx)
print("after: ", typed.schema["order_date"], "|", typed.schema["segment"])
typed.schema

# %%
# Duplicate order rows crept in during export — drop on the business key.
deduped = clean.drop_duplicate_rows(typed, subset=["order_id"])
print(f"{typed.height} rows -> {deduped.height} after de-duplicating on order_id")

# %%
# A few revenue values are wildly inflated — quantify, then winsorize.
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

# %%
# Model-ready categoricals without one-hot explosion: frequency-encode `segment` and lump the
# thinnest `channel` into "other" (rare levels carry too few rows). Target encoding lives in
# `preprocess.make_encoder` (fit on train only).
encoded = compact.pipe(transform.frequency_encode, ["segment"]).pipe(
    transform.group_rare, "channel", min_share=0.15
)
encoded.group_by("channel").agg(pl.len()).sort("len", descending=True)

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
# Persist for downstream notebooks (Parquet, not CSV).
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
# Two ways to tame skew. log1p is the quick stateless move; a fitted power transform
# (Yeo-Johnson / Box-Cox) learns the exponent for max normality — the upgrade when log isn't
# enough, and being fitted it belongs in `preprocess` (train-only). Compare residual skew.
log_revenue = np.log1p(compact["revenue"].to_numpy())
power = preprocess.make_power_transformer(method="yeo-johnson")
yj_revenue = power.fit_transform(compact["revenue"].to_numpy().reshape(-1, 1)).ravel()
for label, series in [
    ("raw", compact["revenue"].to_numpy()),
    ("log1p", log_revenue),
    ("yeo-johnson", yj_revenue),
]:
    print(f"{label:12s} skew {stats.describe_distribution(series)['skew']:+.2f}")

fig, axes = base.grid(2)
eda.qq(yj_revenue, ax=axes[0], title="QQ — Yeo-Johnson(revenue)")
eda.histogram(yj_revenue, ax=axes[1], title="Yeo-Johnson(revenue)")
print("normality (D'Agostino):", stats.normality_test(yj_revenue, method="dagostino"))

# %%
# Univariate fences miss multivariate outliers — normal on each axis but odd in the joint shape.
# Mahalanobis accounts for cross-column correlation; largest distances are the joint anomalies.
numeric_cols = ["revenue", "units", "unit_price", "discount"]
maha = distance.mahalanobis_outliers(compact.select(numeric_cols).to_numpy())
flagged = compact.select(numeric_cols).with_columns(pl.Series("mahalanobis", maha))
print(flagged.sort("mahalanobis", descending=True).head(5))

# %% [markdown]
# ## 6. Explore — relationships across categories

# %%
fig, axes = base.grid(3, ncols=3)
eda.count_bar(compact, "segment", ax=axes[0], title="Orders by segment")
eda.boxplot_by(compact, "revenue", "segment", ax=axes[1], title="Revenue by segment")
eda.crosstab_heatmap(compact, "region", "segment", ax=axes[2], title="Region x Segment")

# %%
sample_df = transform.sample(compact, n=800, seed=42)
eda.pairplot(sample_df, ["revenue", "unit_price", "units", "discount"], hue="segment")

# %% [markdown]
# ## 7. Correlations — Pearson, Spearman, and an interactive heatmap
# Pearson measures *linear* association and bends to outliers; Spearman ranks first, so it reads
# monotonic relationships robustly. Where the two disagree, look at the scatter.

# %%
stats.spearman(compact.select(["revenue", "units", "unit_price", "discount"]))

# %%
# Plotly version — hover, zoom, pan. (Needs the `interactive` extra.)
interactive.correlation_heatmap(compact, title="Numeric correlations (Pearson)")

# %% [markdown]
# **Takeaways:** types fixed and memory cut via downcast; `segment` cleaned + imputed (after a
# missingness-mechanism check said simple fills are safe, with `_missing` flags keeping the
# signal); duplicate orders and revenue outliers handled; rare channels lumped and frequencies
# encoded for modeling; revenue is log-normal-ish and varies by segment. The clean frame is
# cached to `data/processed/` for the feature/KPI notebook.
