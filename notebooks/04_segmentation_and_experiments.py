# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04 · Segmentation, anomalies & experiments
#
# Three "analyze" workflows on the customer base:
# **(1)** unsupervised segmentation (k-means + PCA), **(2)** anomaly detection, and
# **(3)** an A/B test of the retention campaign (`group`) with sample-size/power and a causal check.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()

customers = read_parquet(ROOT / "data" / "raw" / "customers.parquet")
features_num = [
    "age",
    "tenure_months",
    "num_products",
    "sessions_30d",
    "support_tickets",
    "monthly_spend",
    "satisfaction",
]

# Impute + scale once (clustering and PCA need a clean, standardized matrix).
pre = preprocess.make_preprocessor(numeric=features_num, scale=True, impute=True)
features_x = pre.fit_transform(customers.select(features_num).to_pandas())
features_x.shape

# %% [markdown]
# ## 1. How many segments? Elbow + silhouette

# %%
ks = list(range(2, 9))
k_elbow, inertias = segment.elbow_scores(features_x, ks)
k_sil, sils = segment.silhouette_scores(features_x, ks)
fig, axes = base.grid(2)
cluster.elbow(k_elbow, inertias, ax=axes[0], title="Elbow (inertia)")
cluster.silhouette(k_sil, sils, ax=axes[1], title="Silhouette")

# %% [markdown]
# ## 2. Fit k-means, visualize in PCA space

# %%
km = segment.make_clusterer("kmeans", n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(features_x)
coords, evr = segment.pca(features_x, n_components=2)
fig, axes = base.grid(2)
cluster.cluster_scatter(coords, labels, ax=axes[0], title="Customer segments (PCA)")
cluster.explained_variance(evr, ax=axes[1], title="PCA explained variance")

# %%
# Profile the segments — who are they, and which churns most?
customers.with_columns(pl.Series("cluster", labels)).group_by("cluster").agg(
    pl.len().alias("n"),
    pl.col("monthly_spend").mean().round(0).alias("avg_spend"),
    pl.col("tenure_months").mean().round(0).alias("avg_tenure"),
    pl.col("churned").mean().round(3).alias("churn_rate"),
).sort("cluster")

# %% [markdown]
# ## 3. Anomaly detection
# Isolation Forest flags unusual customers (1 = normal, -1 = outlier).

# %%
detector = anomaly.make_detector("isolation_forest", contamination=0.03, random_state=42)
flags = anomaly.anomaly_labels(detector, features_x)
print(f"{int((flags == -1).sum())} anomalies of {len(flags)} ({(flags == -1).mean():.1%})")

# %%
outliers = customers.with_columns(pl.Series("outlier", flags == -1)).filter(pl.col("outlier"))
outliers.select(features_num).describe()

# %% [markdown]
# ## 4. A/B test — did the retention campaign work?
# `group` splits customers into control vs treatment; the campaign aims to *retain* (reduce churn).

# %%
ctrl = customers.filter(pl.col("group") == "control")
trt = customers.filter(pl.col("group") == "treatment")
ctrl_retained = int((ctrl["churned"] == 0).sum())
trt_retained = int((trt["churned"] == 0).sum())
experiment.analyze_conversions(ctrl_retained, ctrl.height, trt_retained, trt.height)

# %%
# A guardrail metric (spend) should be unaffected — compare the means.
spend_ctrl = ctrl["monthly_spend"].drop_nulls().to_numpy()
spend_trt = trt["monthly_spend"].drop_nulls().to_numpy()
experiment.analyze_means(spend_ctrl, spend_trt)

# %%
# Planning: how many per arm to detect a 3pp lift, and the power we actually had.
base_retention = float((ctrl["churned"] == 0).mean())
per_arm = stats.sample_size_proportion(base_retention, base_retention + 0.03)
print(f"sample size needed per arm (3pp lift): {per_arm}")
print(f"power at n={ctrl.height} for a small effect (0.1): {stats.power(0.1, n=ctrl.height):.2f}")

# %% [markdown]
# ## 5. Causal cross-check (observational)
# Uplift on retention, plus a propensity model + matching (here `group` is randomized, so propensity
# is ~0.5 — this is the machinery you'd use when treatment is *not* randomly assigned).

# %%
treat = (customers["group"] == "treatment").to_numpy().astype(int)
retained = (customers["churned"] == 0).to_numpy().astype(int)
print(f"uplift in retention: {causal.uplift(retained[treat == 1], retained[treat == 0]):+.4f}")

ps = causal.propensity_scores(features_x, treat)
matched = causal.match_on_propensity(ps, treat, caliper=0.05)
print(f"propensity range [{ps.min():.2f}, {ps.max():.2f}]; {len(matched)} treated units matched")

# %%
# Difference-in-differences (illustrative pre/post retention for the two arms).
did = causal.difference_in_differences(
    control_before=0.80, control_after=0.79, treat_before=0.80, treat_after=0.86
)
print(f"DiD estimate: {did:+.3f}")

# %% [markdown]
# ## 6. The segments, interactively (Plotly)
# Hover a point to inspect it; the PCA scatter is far more useful when it's interactive.

# %%
pca_df = pl.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels.astype(str)})
interactive.scatter(pca_df, "pc1", "pc2", color="cluster", title="Customer segments (PCA)")

# %% [markdown]
# **Takeaways:** four interpretable customer segments with different churn; a small, profilable
# anomaly set; and an A/B test showing the campaign lifts retention while spend is unaffected —
# backed by power analysis and the causal toolkit.
