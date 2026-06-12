# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 08 · Regression for inference — reading effects you can defend
#
# `modeling` predicts; `analytics.regression` explains. We read spend and churn drivers off
# coefficient tables (OLS, GLM), check the assumptions that make those p-values honest
# (collinearity, heteroscedasticity, residuals), handle panel confounding with fixed effects,
# pool small groups with a mixed model — then cross-check the story with a black-box model and
# model-agnostic diagnostics.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()

customers = (
    read_parquet(ROOT / "data" / "raw" / "customers.parquet")
    .pipe(clean.fill_missing, strategy="median", columns=["monthly_spend", "satisfaction"])
    .to_dummies(columns=["plan"])
)
drivers = [
    "age",
    "tenure_months",
    "num_products",
    "sessions_30d",
    "support_tickets",
    "satisfaction",
]

# %% [markdown]
# ## 1. Multicollinearity first (VIF)
# Two systems report the same money ("billed amount" ≈ monthly spend). VIF >> 10 says the pair is
# unusable together — coefficients would be unstable and uninterpretable. Drop one, then proceed.

# %%
rng = np.random.default_rng(42)
with_dup = customers.with_columns(
    (pl.col("monthly_spend") * 1.02 + rng.normal(0, 5, customers.height)).alias("billed_amount")
)
regression.vif(with_dup.select([*drivers, "monthly_spend", "billed_amount"]))

# %% [markdown]
# ## 2. OLS on spend — and why its assumptions fail here
# Spend is driven by plan; the OLS fit reads that cleanly. But the residual diagnostics flag
# heteroscedasticity (premium variance is far larger), so the *standard errors* can't be trusted
# as-is — the classic case for a log scale or a Gamma GLM.

# %%
x_spend = ["plan_premium", "plan_standard", *drivers]
fit_ols = regression.ols_fit(customers, y="monthly_spend", x=x_spend)
print(f"R² {fit_ols.r_squared:.3f}   AIC {fit_ols.aic:,.0f}   n {fit_ols.n}")
fit_ols.coefficients

# %%
betas = fit_ols.coefficients["coef"].to_numpy()
features = customers.select(x_spend)
predicted = betas[0] + features.to_numpy() @ betas[1:]
residuals = customers["monthly_spend"].to_numpy() - predicted
print(regression.linear_assumptions(features, residuals))

# %%
fig, axes = base.grid(2)
model.residuals(
    customers["monthly_spend"].to_numpy(),
    predicted,
    ax=axes[0],
    title="Residuals — funnel = heteroscedastic",
)
eda.qq(residuals, ax=axes[1], title="Residual Q-Q")

# %% [markdown]
# ## 3. The fix: model spend on the right scale (Gamma GLM, log link)
# exp(coef) reads as a *multiplier* on spend: premium customers spend ~4-5x the basic baseline,
# matching how the data was generated.

# %%
fit_gamma = regression.glm_fit(customers, y="monthly_spend", x=x_spend, family="gamma")
fit_gamma.coefficients.with_columns(pl.col("coef").exp().round(2).alias("multiplier"))

# %% [markdown]
# ## 4. Churn drivers — logistic GLM (log-odds you can quote)
# Negative coefficients protect (tenure, sessions, satisfaction, spend); tickets and the basic
# plan push churn up. exp(coef) = odds ratio per unit.

# %%
x_churn = [*drivers, "plan_basic"]
fit_churn = regression.glm_fit(customers, y="churned", x=x_churn, family="binomial")
fit_churn.coefficients.with_columns(pl.col("coef").exp().round(3).alias("odds_ratio"))

# %%
# Interaction check: does satisfaction buffer the effect of support tickets? (Moderation test —
# here the interaction is ~0: ticket pain doesn't depend on satisfaction level in this data.)
with_ix = customers.pipe(transform.add_interactions, [("support_tickets", "satisfaction")])
fit_ix = regression.glm_fit(
    with_ix, y="churned", x=[*x_churn, "support_tickets_x_satisfaction"], family="binomial"
)
fit_ix.coefficients.filter(pl.col("term") == "support_tickets_x_satisfaction")

# %% [markdown]
# ## 5. Panel data — fixed vs mixed effects (synthetic stores, known truth)
# 24 stores, 26 weeks. Store quality drives both promo intensity and sales (a classic entity-level
# confounder), and the *true* within-store promo effect is **+2.0**. Pooled OLS is biased upward;
# fixed effects demean it away; the mixed model adds partial pooling and reports the between-store
# variance.

# %%
n_stores, n_weeks = 24, 26
store = np.repeat(np.arange(n_stores), n_weeks)
quality = rng.normal(0.0, 2.0, n_stores)[store]  # unobserved store quality
promo = 0.6 * quality + rng.normal(0.0, 1.0, store.size)
sales = 100.0 + 2.0 * promo + 8.0 * quality + rng.normal(0.0, 2.0, store.size)
panel = pl.DataFrame({"store": store, "promo": promo, "sales": sales})

pooled = regression.ols_fit(panel, y="sales", x=["promo"])
within = regression.fixed_effects(panel, y="sales", x=["promo"], entity="store")
mixed = regression.mixed_effects(panel, y="sales", x=["promo"], group="store")

pooled_coef = pooled.coefficients.filter(pl.col("term") == "promo")["coef"][0]
within_coef = within.coefficients.filter(pl.col("term") == "promo")["coef"][0]
mixed_coef = mixed.coefficients.filter(pl.col("term") == "promo")["coef"][0]
print("true effect      +2.00")
print(f"pooled OLS       {pooled_coef:+.2f}   (confounded by store quality)")
print(f"fixed effects    {within_coef:+.2f}   (within-store, confounder absorbed)")
print(f"mixed effects    {mixed_coef:+.2f}   (group variance {mixed.group_variance:.1f})")

# %% [markdown]
# ## 6. Cross-check with a black box — do the stories agree?
# A gradient-boosted model + permutation importance (held-out data) should rank the same churn
# drivers the GLM found; the learning curve says whether more data would help; RFECV asks how many
# features earn their keep.

# %%
train_df, test_df = split.train_test_split(customers, test_size=0.25, stratify="churned", seed=42)
booster = train.fit(
    registry.make_model("hist_gradient_boosting", task="classification", random_state=42),
    train_df.select(x_churn),
    train_df["churned"],
)
importance = evaluate.permutation_importance(
    booster, test_df.select(x_churn), test_df["churned"], scoring="roc_auc", seed=42
)
importance

# %%
fig, axes = base.grid(1, ncols=1)
explain.permutation_importance(
    importance["feature"].to_list(),
    importance["importance_mean"].to_numpy(),
    importance["importance_std"].to_numpy(),
    ax=axes[0],
    title="Permutation importance (held-out AUC drop)",
)

# %%
sizes, train_scores, val_scores = evaluate.learning_curve_scores(
    registry.make_model("hist_gradient_boosting", task="classification", random_state=42),
    customers.select(x_churn),
    customers["churned"],
    cv=3,
    scoring="roc_auc",
)
selection = evaluate.rfecv_scores(
    registry.make_model("logistic", task="classification", max_iter=2000),
    customers.select(x_churn),
    customers["churned"],
    cv=3,
    scoring="roc_auc",
)
fig, axes = base.grid(2)
model.learning_curve(
    sizes, train_scores, val_scores, ax=axes[0], title="Learning curve (bias vs variance)"
)
model.feature_selection_curve(
    selection.n_features,
    selection.mean_scores,
    std=selection.std_scores,
    ax=axes[1],
    title="RFECV — features that earn their keep",
)
print(f"RFECV keeps: {selection.selected}")

# %% [markdown]
# **Takeaways:** plan dominates spend and the Gamma GLM is the honest scale for it (OLS residuals
# fail Breusch-Pagan); churn falls with tenure/sessions/satisfaction/spend and rises with tickets
# and the basic plan — the boosted model's permutation importance ranks the same drivers, the
# learning curve is flat-ish (more data won't move much), and RFECV confirms a compact driver set.
# On panels, demand the fixed-effects estimate before believing a pooled slope.
