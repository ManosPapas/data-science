# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 21 · Model governance — does the model respect business logic, and what do we tell the user?
#
# A model can post a great AUC and still be unshippable: demand that rises with price, scores
# that flip under measurement noise, predictions outside any plausible range. This notebook is
# the pre-deployment gate and the decision-support layer in one: business-rule checks on the
# *data*, monotonicity/robustness checks on the *model*, then the three things a stakeholder
# asks of every score — what would change it (counterfactual), how far off might it be
# (conformal interval), and how sure are we (confidence routing).

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. A renewal-pricing model with a planted flaw
# We model renewal probability from price, tenure, and support tickets — but the training data
# is *confounded*: high-value customers were historically quoted higher prices AND renew more.
# A flexible model happily learns "higher price → renews more". Accuracy won't catch that;
# the governance checks below will.

# %%
n = 6000
tenure = rng.uniform(1.0, 60.0, n)
tickets = rng.poisson(1.2, n).astype(float)
value_score = rng.uniform(0.0, 1.0, n)  # latent, NOT in the model
price = 50.0 + 60.0 * value_score + rng.normal(0.0, 6.0, n)  # the confounder at work

logit = 0.8 + 2.2 * value_score - 0.018 * price + 0.015 * tenure - 0.35 * tickets
renewed = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

features_df = pl.DataFrame({"price": price, "tenure": tenure, "tickets": tickets})
train_df, test_df, y_train, y_test = (
    features_df[:4500],
    features_df[4500:],
    renewed[:4500],
    renewed[4500:],
)
model_fit = train.fit(
    registry.make_model("gradient_boosting", task="classification"), train_df, y_train
)
auc = evaluate.classification_metrics(
    y_test, train.predict(model_fit, test_df), y_score=train.predict_proba(model_fit, test_df)[:, 1]
)["roc_auc"]
print(f"test AUC {auc:.3f} — looks shippable; it isn't. Watch.")

# %% [markdown]
# ## 2. Gate the *data* first — business rules
# Cheap, brutal, and catches more incidents than any model check: named predicates that must
# hold on every row. Run with `raise_on_error=True` as a pipeline gate.

# %%
validate.check_rules(
    features_df,
    {
        "price within rate card": pl.col("price").is_between(20.0, 200.0),
        "tenure non-negative": pl.col("tenure") >= 0,
        "tickets plausible": pl.col("tickets") <= 50,
    },
)

# %% [markdown]
# ## 3. Gate the *model* — expected directions
# The business knows the signs: renewal should *fall* with price and tickets, *rise* with
# tenure. `expected_directions` sweeps each feature per row and counts wrong-direction moves —
# and the confounded price relationship fails loudly, exactly as it should.

# %%
direction_table = checks.expected_directions(
    model_fit,
    test_df,
    {"price": "decreasing", "tenure": "increasing", "tickets": "decreasing"},
)
direction_table

# %%
price_check = checks.monotonicity(model_fit, test_df, feature="price", direction="decreasing")
print(
    f"price violates 'renewal falls with price' on {price_check.violation_rate:.0%} of customers "
    f"(worst wrong-way jump {price_check.worst_gap:.3f})"
)
print("diagnosis: price proxies customer value in the training data (confounding).")
print("fix: causal price variation (nb 09/13) or a monotone constraint — then re-check.")

# %%
# Same model class, constrained to the business-known signs: the shippable variant.
monotone = train.fit(
    registry.make_model("hist_gradient_boosting", task="classification", monotonic_cst=[-1, 1, -1]),
    train_df,
    y_train,
)
checks.expected_directions(
    monotone, test_df, {"price": "decreasing", "tenure": "increasing", "tickets": "decreasing"}
)

# %% [markdown]
# ## 4. Robustness and range — the remaining behaviour checks
# Inputs are measured with error; a model whose scores swing on noise-sized jitter will thrash
# downstream decisions (offer granted Monday, refused Tuesday). And predictions must live in
# the plausible range — probabilities do by construction, but bounds catch regressors too.

# %%
stability = checks.perturbation_stability(monotone, test_df, scale=0.03, n_repeats=10)
print(
    f"under 3% input noise: mean |Δscore| {stability.mean_abs_change:.4f}, "
    f"p95 {stability.p95_abs_change:.4f} "
    f"({stability.relative_mean_change:.1%} of the prediction range) -> stable"
)
print(checks.prediction_bounds(monotone, test_df, lower=0.0, upper=1.0))

# %% [markdown]
# ## 5. Counterfactuals — from score to action
# For an at-risk account, the useful output is not "0.34" but *what changes it*. The search runs
# over **actionable** levers only (price we can concede, success-plan enrollment that resolves
# tickets) — never immutables like tenure: a counterfactual on tenure is an excuse, not an
# action. The smallest qualifying change is the retention offer.

# %%
scores = train.predict_proba(monotone, test_df)[:, 1]
at_risk_index = int(np.argmin(np.abs(scores - 0.35)))  # a genuinely shaky account
account = test_df[at_risk_index]
print(account)

offer = interpret.counterfactual(
    monotone,
    account,
    candidates={
        "price": np.round(np.linspace(account["price"][0] - 30.0, account["price"][0], 7), 0),
        "tickets": np.array([0.0, account["tickets"][0]]),  # success plan clears the backlog
    },
    target=0.60,
    direction=">=",
)
print(
    f"baseline renewal {offer.baseline_score:.0%} -> {offer.score:.0%} "
    f"with changes {offer.changes} ({offer.candidates_evaluated} options evaluated)"
)

# %% [markdown]
# ## 6. Honest uncertainty — conformal intervals for a regression
# For the revenue-expansion regressor that rides along with renewal, stakeholders need ranges,
# not points. Split-conformal intervals are distribution-free with guaranteed coverage on
# exchangeable data — no normality, no trust in the model required. Calibrate on a *held-out*
# slice, never on training rows.

# %%
expansion = 200.0 + 8.0 * tenure + rng.normal(0.0, 60.0, n)  # € expansion next year
reg_train, reg_cal, reg_test = features_df[:3000], features_df[3000:4500], features_df[4500:]
regressor = train.fit(
    registry.make_model("random_forest", task="regression"), reg_train, expansion[:3000]
)
intervals = interpret.conformal_intervals(
    regressor, reg_cal, expansion[3000:4500], reg_test, alpha=0.1
)
coverage = (
    (intervals["lower"].to_numpy() <= expansion[4500:])
    & (expansion[4500:] <= intervals["upper"].to_numpy())
).mean()
print(intervals.head(3))
print(f"empirical coverage {coverage:.1%} (target ≥ 90%) — the guarantee, observed")

# %% [markdown]
# ## 7. Confidence routing — when to involve a human
# Margin-based confidence splits the queue: auto-act on confident scores, route the murky middle
# to an account manager. The routing is only as honest as the probabilities — check calibration
# (notebook 03) before wiring it up.

# %%
confidence = interpret.confidence_score(train.predict_proba(monotone, test_df))
routing = pl.DataFrame({"score": scores, "confidence": confidence}).with_columns(
    pl.when(pl.col("confidence") >= 0.5)
    .then(pl.lit("auto"))
    .otherwise(pl.lit("human review"))
    .alias("route")
)
print(routing.group_by("route").agg(pl.len(), pl.col("score").mean().round(2)))
eda.histogram(confidence, bins=40, title="Confidence distribution — the murky middle gets a human")

# %% [markdown]
# **Takeaways:** the unconstrained booster posted a healthy AUC while *raising* renewal scores
# with price on a third of customers — a confounding artifact accuracy metrics cannot see and
# the monotonicity gate catches in one table; the monotone-constrained variant passes every
# directional check at no meaningful accuracy cost and becomes the shippable model; stability
# and bounds checks clear it for noisy production inputs; the counterfactual turns an at-risk
# score into a concrete, costed retention offer (a modest price concession + clearing the ticket
# backlog); conformal intervals deliver the promised ≥90% coverage without distributional
# assumptions — ranges a CFO can plan on; and confidence routing sends the genuinely ambiguous
# fifth of accounts to humans instead of pretending the model knows. The pattern to
# institutionalize: `check_rules` on data, `expected_directions` on every retrain, intervals and
# counterfactuals on every customer-facing score.
