# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 07 · Statistical inference — how sure are we, and what actually drives the differences?
#
# The stats toolbox on the customer base: describe a skewed metric and fit it a distribution
# (MLE), put honest intervals on means/medians/rates (t, Wilson, bootstrap), test group
# differences three ways (parametric, rank, permutation) with effect sizes and power, triage
# missingness (MCAR vs MAR), catch a Simpson's-paradox reversal, and rank drivers with
# information theory.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()

customers = read_parquet(ROOT / "data" / "raw" / "customers.parquet")
spend = customers["monthly_spend"].drop_nulls().to_numpy()
stats.summary(customers)

# %% [markdown]
# ## 1. What shape is monthly spend? (distributions & MLE)
# Heavily right-skewed — the mean alone misleads. `best_distribution` fits candidate families by
# maximum likelihood and ranks them by AIC; the winner gives us a parametric model for spend.

# %%
print(stats.describe_distribution(spend))
stats.best_distribution(spend, candidates=("norm", "lognorm", "gamma", "expon"))

# %%
fig, axes = base.grid(2)
eda.histogram(spend, ax=axes[0], title="Monthly spend")
eda.qq(spend, ax=axes[1], title="Q-Q vs normal — heavy right tail")
print(stats.normality_test(spend[:500]))  # Shapiro on a subsample: decisively non-normal

# %% [markdown]
# ## 2. Intervals — mean (t), median (bootstrap), churn rate (Wilson)
# Three parameters, three correct interval tools. The bootstrap needs no formula: it resamples
# and reads the interval off the empirical distribution — handy for the median of a skewed metric.

# %%
mean_est, mean_lo, mean_hi = stats.mean_confidence_interval(spend)
print(f"mean spend      {mean_est:7.1f}  [{mean_lo:.1f}, {mean_hi:.1f}]  (t interval, CLT)")

median_est, median_lo, median_hi = stats.bootstrap_ci(
    spend, np.median, n_resamples=2000, method="percentile", seed=42
)
print(f"median spend    {median_est:7.1f}  [{median_lo:.1f}, {median_hi:.1f}]  (bootstrap)")

churn_rate, churn_lo, churn_hi = stats.proportion_confidence_interval(
    int(customers["churned"].sum()), customers.height
)
print(f"churn rate      {churn_rate:7.3f}  [{churn_lo:.3f}, {churn_hi:.3f}]  (Wilson)")

# %% [markdown]
# ## 3. Do premium customers spend more than basic? Test it three ways
# Welch t (means, assumes rough normality of means), Mann-Whitney (ranks), and a permutation test
# (means, no assumptions — shuffles the labels). When all three agree, the conclusion is robust;
# the effect size says whether it *matters*.

# %%
premium = customers.filter(pl.col("plan") == "premium")["monthly_spend"].drop_nulls().to_numpy()
basic = customers.filter(pl.col("plan") == "basic")["monthly_spend"].drop_nulls().to_numpy()

print(f"welch t      {stats.welch_t_test(premium, basic)}")
print(f"mann-whitney {stats.mann_whitney(premium, basic)}")
print(f"permutation  {stats.permutation_test(premium, basic, n_resamples=2000)}")
print(f"cohen's d    {stats.cohens_d(premium, basic):.2f}   (>0.8 = large)")
print(f"cliff's delta {stats.cliffs_delta(premium, basic):.2f}  (rank-based, robust)")

# %%
# Across all three plans in one call — picks the right test and reports the effect size.
print(
    stats.compare_groups(customers.drop_nulls("monthly_spend"), value="monthly_spend", group="plan")
)
stats.group_summary(customers.drop_nulls("monthly_spend"), value="monthly_spend", group="plan")

# %%
# Is churn associated with plan at all? (two categoricals -> chi-square)
print(stats.chi_square(customers["plan"].to_numpy(), customers["churned"].to_numpy()))

# %% [markdown]
# ## 4. Power — design before you test
# `alpha` is the Type I (false-positive) rate we accept; power = 1 - β keeps Type II (missed
# effects) low. Underpowered tests mostly produce "inconclusive" — size the sample first.

# %%
print(f"n per arm to detect d=0.2 at 80% power: {stats.sample_size_mean(0.2)}")
print(f"n per arm to detect churn 15% -> 12%:    {stats.sample_size_proportion(0.15, 0.12)}")
print(f"power for d=0.2 at n=500 per arm:        {stats.power(0.2, n=500):.2f}")

# %% [markdown]
# ## 5. Missingness — MCAR or MAR?
# Before imputing, ask whether missingness *depends* on other columns. Small p-values = MAR
# (impute conditionally, keep a flag); all large = consistent with MCAR (simple fills are safe).

# %%
stats.missingness(customers).head(5)

# %%
stats.missingness_dependence(
    customers.select(
        [
            "monthly_spend",
            "age",
            "tenure_months",
            "sessions_30d",
            "support_tickets",
            "plan",
            "region",
        ]
    ),
    "monthly_spend",
)

# %%
# All p-values are large -> MCAR here, so flag + median-fill is defensible.
filled = customers.pipe(clean.add_missing_indicators).pipe(
    clean.fill_missing, strategy="median", columns=["monthly_spend", "satisfaction"]
)
filled.select(["monthly_spend", "monthly_spend_missing"]).head(4)

# %% [markdown]
# ## 6. Simpson's paradox — when the pooled trend lies
# Synthetic promo data (seeded): *within* each tier, deeper discounts go with lower order value;
# but premium customers get both deeper discounts *and* higher order values, so the pooled slope
# flips positive. `simpsons_check` flags the reversal.

# %%
rng = np.random.default_rng(42)
n = 400
depth_basic = rng.uniform(0.00, 0.10, n)
depth_premium = rng.uniform(0.10, 0.25, n)
promo = pl.DataFrame(
    {
        "discount_depth": np.concatenate([depth_basic, depth_premium]),
        "order_value": np.concatenate(
            [
                90 - 150 * depth_basic + rng.normal(0, 4, n),
                260 - 150 * depth_premium + rng.normal(0, 4, n),
            ]
        ),
        "tier": ["basic"] * n + ["premium"] * n,
    }
)
check = stats.simpsons_check(promo, x="discount_depth", y="order_value", group="tier")
print(f"pooled slope {check['overall_slope']:+.0f}, within-tier slope {check['within_slope']:+.0f}")
print(f"reversal: {check['reversal']} -> report the within-tier effect")
check["by_group"]

# %% [markdown]
# ## 7. Information theory — which categorical tells us most about churn?
# Information gain = the entropy of `churned` removed by knowing the feature; KL divergence
# measures how different the plan mix of churners is from non-churners.

# %%
print(f"entropy of churned: {stats.entropy(customers['churned'].to_numpy()):.3f} bits")
for feature in ("plan", "segment", "region"):
    gain = stats.information_gain(customers, feature, "churned")
    print(f"information gain from {feature:8s}: {gain:.4f} bits")

# %%
plan_mix = (
    customers.group_by("churned", "plan")
    .len()
    .pivot("plan", index="churned", values="len")
    .sort("churned")
)
shares = plan_mix.select(["basic", "premium", "standard"]).to_numpy().astype(float)
print(f"KL(churned plan mix vs retained): {stats.kl_divergence(shares[1], shares[0]):.4f} bits")

# %%
numeric_drivers = [
    "age",
    "tenure_months",
    "num_products",
    "sessions_30d",
    "support_tickets",
    "monthly_spend",
    "satisfaction",
    "churned",
]
stats.mutual_information(customers.select(numeric_drivers), "churned", task="classification")

# %% [markdown]
# **Takeaways:** spend is gamma-like (model it on a log/GLM scale, quote medians with bootstrap
# intervals); the premium-vs-basic spend gap is large and survives all three tests; spend
# missingness looks MCAR, so simple imputation is safe; the promo example shows why pooled slopes
# need a Simpson's check; and tenure/tickets/satisfaction carry the most information about churn —
# the shortlist for the regression notebook (08).
