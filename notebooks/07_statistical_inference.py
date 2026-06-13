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
# The stats toolbox on the customer base: describe a skewed metric, fit it a distribution by
# maximum likelihood and *use* the fitted model, put honest intervals on means/medians/quantiles/
# rates (t, bootstrap, Wilson), test group differences three ways with effect sizes and power,
# triage missingness (MCAR vs MAR), catch a Simpson's-paradox reversal, and rank drivers with
# information theory. (Profiling and cleaning live in notebook 01; this one is about *inference*.)

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
# maximum likelihood and ranks them by AIC; it also reports **BIC** (the heavier, n-scaled penalty
# ≈ minimum description length — it prefers simpler families as data grows). When AIC and BIC agree
# on the winner the choice is robust; the KS column scores absolute fit (best-of-a-bad-bunch is
# still bad).

# %%
print(stats.describe_distribution(spend))
stats.best_distribution(spend, candidates=("norm", "lognorm", "gamma", "expon", "weibull_min"))

# %%
fig, axes = base.grid(2)
eda.histogram(spend, ax=axes[0], title="Monthly spend")
eda.qq(spend, ax=axes[1], title="Q-Q vs normal — heavy right tail")
print(stats.normality_test(spend[:500]))  # Shapiro on a subsample: decisively non-normal

# %%
# The payoff of a fitted model: parametric tail probabilities. Fit the gamma by MLE, simulate
# from it, and ask business questions the histogram answers only noisily in the tail.
fit = stats.fit_distribution(spend, "gamma")
shape, loc, scale = fit["params"]
rng = np.random.default_rng(42)
simulated = rng.gamma(shape, scale, 200_000) + loc
for cutoff in (400, 800):
    in_model = float((simulated > cutoff).mean())
    observed = float((spend > cutoff).mean())
    print(f"P(spend > {cutoff})  model {in_model:.3f}  vs  empirical {observed:.3f}")

# %% [markdown]
# ## 2. Intervals — mean (t), median & p90 (bootstrap), churn rate (Wilson)
# Four parameters, the right interval tool for each. The bootstrap needs no formula: it resamples
# and reads the interval off the empirical distribution — which is the only honest option for the
# median and the 90th percentile of a skewed metric.

# %%
mean_est, mean_lo, mean_hi = stats.mean_confidence_interval(spend)
print(f"mean spend      {mean_est:7.1f}  [{mean_lo:.1f}, {mean_hi:.1f}]  (t interval, CLT)")

median_est, median_lo, median_hi = stats.bootstrap_ci(
    spend, np.median, n_resamples=2000, method="percentile", seed=42
)
print(f"median spend    {median_est:7.1f}  [{median_lo:.1f}, {median_hi:.1f}]  (bootstrap)")

p90_est, p90_lo, p90_hi = stats.bootstrap_ci(
    spend,
    lambda sample: float(np.percentile(sample, 90)),
    n_resamples=1000,
    method="percentile",
    seed=42,
)
print(f"p90 spend       {p90_est:7.1f}  [{p90_lo:.1f}, {p90_hi:.1f}]  (bootstrap)")

churn_rate, churn_lo, churn_hi = stats.proportion_confidence_interval(
    int(customers["churned"].sum()), customers.height
)
print(f"churn rate      {churn_rate:7.3f}  [{churn_lo:.3f}, {churn_hi:.3f}]  (Wilson)")

# %% [markdown]
# ## 3. Do premium customers spend more than basic? Test it three ways
# Welch t (means, assumes rough normality of means), Mann-Whitney (ranks), and a permutation test
# (means, no assumptions — shuffles the labels). When all three agree, the conclusion is robust.
# Then the question that actually matters commercially: *how big* is the difference — Cohen's d
# (with the small-sample Hedges correction) and the rank-based Cliff's delta.

# %%
premium = customers.filter(pl.col("plan") == "premium")["monthly_spend"].drop_nulls().to_numpy()
basic = customers.filter(pl.col("plan") == "basic")["monthly_spend"].drop_nulls().to_numpy()

print(f"welch t       {stats.welch_t_test(premium, basic)}")
print(f"mann-whitney  {stats.mann_whitney(premium, basic)}")
print(f"permutation   {stats.permutation_test(premium, basic, n_resamples=2000)}")
print(f"cohen's d     {stats.cohens_d(premium, basic):.2f}   (>0.8 = large)")
print(f"hedges' g     {stats.hedges_g(premium, basic):.2f}   (bias-corrected d)")
print(f"cliff's delta {stats.cliffs_delta(premium, basic):.2f}   (rank-based, robust)")

# %%
fig, axes = base.grid(1, ncols=1)
eda.ks(premium, basic, labels=("premium", "basic"), ax=axes[0], title="Spend ECDFs by plan")

# %%
# Across all three plans in one call — picks the right test and reports the effect size; eta²
# is the share of spend variance that plan membership alone explains.
plans = [
    customers.filter(pl.col("plan") == p)["monthly_spend"].drop_nulls().to_numpy()
    for p in ("basic", "standard", "premium")
]
print(
    stats.compare_groups(customers.drop_nulls("monthly_spend"), value="monthly_spend", group="plan")
)
print(f"eta² (variance explained by plan): {stats.eta_squared(plans):.2f}")
stats.group_summary(customers.drop_nulls("monthly_spend"), value="monthly_spend", group="plan")

# %%
# Is churn associated with plan at all? (two categoricals -> chi-square)
print(stats.chi_square(customers["plan"].to_numpy(), customers["churned"].to_numpy()))

# %% [markdown]
# ## 3b. The rest of the test battery — pick the test by the question
# One-sample (mean vs a target), exact (small 2x2 where chi-square's >=5 rule fails),
# goodness-of-fit (does a category mix match an expectation?), and the *paired/repeated* tests for
# the same units measured more than once. Reaching for the right one is the skill; the wrong test
# on the right data is still the wrong answer.

# %%
# One-sample: is mean satisfaction different from the 7.0 target the CX team commits to?
satisfaction = customers["satisfaction"].drop_nulls().to_numpy()
print(f"one-sample t (vs 7.0)   {stats.one_sample_t_test(satisfaction, 7.0)}")

# Goodness-of-fit: is the region mix uniform, or skewed toward some regions?
region_counts = customers["region"].value_counts(sort=True)["count"].to_numpy()
print(f"chi-square GOF (uniform){stats.chi_square_gof(region_counts)}")

# Fisher's exact: a rare-cell 2x2 (a tiny pilot cohort) where chi-square is untrustworthy
pilot = [[8, 2], [3, 20]]  # converted/not by arm, small n
print(f"fisher's exact          {stats.fishers_exact(pilot)}  (statistic = odds ratio)")

# %%
# Paired / repeated measures: the SAME customers surveyed under three onboarding variants.
# Friedman = the non-parametric repeated-measures ANOVA; Wilcoxon signed-rank = its paired case.
base_score = rng.normal(6.5, 1.2, 200)
variant_a = base_score + rng.normal(0.0, 0.4, 200)  # ~ no change
variant_b = base_score + 0.6 + rng.normal(0.0, 0.4, 200)  # a real lift
variant_c = base_score + 1.1 + rng.normal(0.0, 0.4, 200)  # a bigger lift
print(f"friedman (3 variants)   {stats.friedman_test(variant_a, variant_b, variant_c)}")
print(f"wilcoxon (a vs c)       {stats.wilcoxon_signed_rank(variant_a, variant_c)}")

# %% [markdown]
# ## 4. Correlation — pick the coefficient by the variable types
# Pearson is the *wrong* default for anything but two continuous, linear variables. The right
# coefficient depends on the pair of types — `correlation_kind` is the chooser, and the suite
# below covers each case on real customer columns.

# %%
# Continuous ↔ continuous, but curved: Pearson under-reads a monotonic relationship; Spearman and
# Kendall (rank-based, robust to the curve and to ties) nail it.
curved_x = rng.uniform(1.0, 10.0, 300)
curved_y = curved_x**4 + rng.normal(0.0, 80.0, 300)
for method in ("pearson", "spearman", "kendall"):
    print(f"{method:9s} {stats.correlation_test(curved_x, curved_y, method=method)}")

# %%
# The chooser maps variable types -> the right function, so the call site is never guesswork.
for x_kind, y_kind in [
    ("continuous", "continuous"),
    ("binary", "continuous"),
    ("nominal", "nominal"),
    ("binary", "binary"),
    ("ordinal", "continuous"),
]:
    print(f"{x_kind:11s} x {y_kind:11s} -> {stats.correlation_kind(x_kind, y_kind)}")

# %%
# Binary vs continuous: does churn (0/1) relate to monthly spend? -> point-biserial.
churned = customers["churned"].to_numpy()
spend_filled = customers["monthly_spend"].fill_null(strategy="mean").to_numpy()
print(f"point-biserial (churn vs spend)  {stats.point_biserial(churned, spend_filled)}")

# Categorical vs categorical: plan-region association magnitude -> Cramér's V (+ chi-square p).
cramers = stats.cramers_v(customers["plan"].to_numpy(), customers["region"].to_numpy())
print(f"Cramer's V  (plan vs region)     {cramers:.3f}")

# Binary vs binary: churn vs is-premium -> phi (signed).
is_premium = (customers["plan"] == "premium").cast(pl.Int8).to_numpy()
print(f"phi         (churn vs is_premium) {stats.phi_coefficient(churned, is_premium):+.3f}")

# %%
# Partial correlation: spend and sessions look related — but tenure drives both. Control for it.
raw = stats.correlation_test(
    customers["monthly_spend"].fill_null(strategy="mean").to_numpy(),
    customers["sessions_30d"].to_numpy(),
)
partial = stats.partial_correlation(
    customers.with_columns(pl.col("monthly_spend").fill_null(strategy="mean")),
    x="monthly_spend",
    y="sessions_30d",
    covariates=["tenure_months"],
)
print(
    f"spend vs sessions  raw r={raw.statistic:+.3f}   "
    f"partial (control tenure) r={partial.statistic:+.3f}"
)

# %% [markdown]
# ## 5. Power — design before you test
# `alpha` is the Type I (false-positive) rate we accept; power = 1 - β keeps Type II (missed
# effects) low. Underpowered tests mostly produce "inconclusive" — size the sample first.

# %%
print(f"n per arm to detect d=0.2 at 80% power: {stats.sample_size_mean(0.2)}")
print(f"n per arm to detect churn 15% -> 12%:    {stats.sample_size_proportion(0.15, 0.12)}")
print(f"power for d=0.2 at n=500 per arm:        {stats.power(0.2, n=500):.2f}")

# %% [markdown]
# ## 6. Missingness — MCAR or MAR?
# Before imputing, ask whether missingness *depends* on other columns. Small p-values = MAR
# (impute conditionally — `preprocess.make_imputer("knn"/"iterative")` — and keep a flag); all
# large = consistent with MCAR (simple fills are safe).

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
# ## 7. Simpson's paradox — when the pooled trend lies
# Synthetic promo data (seeded): *within* each tier, deeper discounts go with lower order value;
# but premium customers get both deeper discounts *and* higher order values, so the pooled slope
# flips positive. `simpsons_check` flags the reversal; which slope to report is a causal question
# (notebook 09's DAG section is exactly this decision).

# %%
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

# %%
fig, axes = base.grid(1, ncols=1)
eda.scatter(promo, "discount_depth", "order_value", hue="tier")

# %% [markdown]
# ## 8. Information theory — which categorical tells us most about churn?
# Information gain = the entropy of `churned` removed by knowing the feature (exactly what a
# decision tree maximizes per split); KL divergence measures how different the plan mix of
# churners is from non-churners; mutual information does the same job for numeric features.

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
# **Takeaways:** spend is gamma-like (AIC and BIC agree) — quote medians/p90s with bootstrap
# intervals and use the fitted gamma for tail probabilities; the premium-vs-basic gap is large
# (d ≈ 2, plan explains ~a third of spend variance) and survives all three tests; the test you
# reach for is set by the question (one-sample vs target, Fisher for tiny cells, Friedman/Wilcoxon
# for repeated measures); correlation coefficient is set by the *variable types* — point-biserial,
# Cramér's V, phi, and a partial correlation each answer a different pair, and Pearson is the wrong
# default for most of them; spend missingness looks MCAR, so simple imputation is safe; pooled
# slopes need a Simpson's check; and tenure/tickets/satisfaction carry the most information about
# churn — the shortlist notebook 08 quantifies with confidence intervals.
