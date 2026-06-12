# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 09 · Causal inference — estimating effects when you can't (fully) randomize
#
# Every estimator here is run on **simulated data with a known true effect**, so you can see each
# one recover (or miss) the truth. The sequence mirrors real decisions: a confounded voucher
# (matching/weighting), a regional rollout (DiD), an opt-in campaign (ITT/TOT), confounded
# discounts (IV), a score threshold (RDD), a single-market launch (synthetic control), and
# "who should we target?" (subgroups + uplift modeling).

# %%
from core.config import ROOT  # noqa: F401  (kept for parity with the other notebooks)
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 0. Draw the problem first (DAG)
# Loyalty drives both voucher take-up and spend — a confounder. The backdoor path
# voucher ← loyalty → spend must be blocked (condition on loyalty); never condition on a mediator
# or collider.

# %%
conceptual.dag(
    [("loyalty", "voucher"), ("loyalty", "spend"), ("voucher", "spend")],
    title="Voucher → spend, confounded by loyalty",
)

# %% [markdown]
# ## 1. Confounded voucher — naive vs matching vs weighting (true effect **+10**)
# Loyal customers grab the voucher *and* spend more anyway, so the naive comparison overstates
# the effect. Both propensity tools recover it: matching compares like with like; IPW reweights
# everyone to the same loyalty mix.

# %%
n = 4000
loyalty = rng.normal(0.0, 1.0, n)
voucher = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * loyalty))).astype(int)
spend = 100.0 + 10.0 * voucher + 25.0 * loyalty + rng.normal(0.0, 10.0, n)

naive = causal.uplift(spend[voucher == 1], spend[voucher == 0])
propensity = causal.propensity_scores(loyalty.reshape(-1, 1), voucher)
weighted = causal.ipw_ate(spend, voucher, propensity)

matches = causal.match_on_propensity(propensity, voucher, caliper=0.02)
treated_idx = np.where(voucher == 1)[0][matches >= 0]
matched_effect = float(np.mean(spend[treated_idx] - spend[matches[matches >= 0]]))

print("true effect          +10.0")
print(f"naive difference     {naive:+.1f}   (loyalty bias included)")
print(f"IPW ATE              {weighted:+.1f}")
print(f"matched ATT          {matched_effect:+.1f}   ({(matches >= 0).sum()} pairs within caliper)")

# %%
# The assumption both tools lean on is *positivity*: every unit must have a real chance of either
# treatment. Overlapping propensity distributions = comparable units exist at every score; a
# separated pair would mean some customers are never comparable and no reweighting can fix that.
fig, axes = base.grid(1, ncols=1)
model.score_distribution(voucher, propensity, ax=axes[0], title="Propensity overlap (positivity)")

# %% [markdown]
# ## 2. Regional rollout — difference-in-differences (true effect **+8**)
# One region gets the campaign; both share a market-wide +5 trend. DiD subtracts the control's
# change, removing any time-invariant gap between the regions.

# %%
weeks = 30
common_trend = np.linspace(0.0, 5.0, 2 * weeks)
control_sales = 200.0 + common_trend + rng.normal(0.0, 2.0, 2 * weeks)
treated_sales = 215.0 + common_trend + rng.normal(0.0, 2.0, 2 * weeks)
treated_sales[weeks:] += 8.0  # campaign starts at week 30 in the treated region

did = causal.difference_in_differences(
    control_before=float(control_sales[:weeks].mean()),
    control_after=float(control_sales[weeks:].mean()),
    treat_before=float(treated_sales[:weeks].mean()),
    treat_after=float(treated_sales[weeks:].mean()),
)
print(f"DiD estimate: {did:+.1f}   (true +8; the level gap and common trend cancel)")

# %% [markdown]
# ## 3. Opt-in campaign — ITT vs TOT (true effect on the treated **+6**)
# We *assign* the email to half the base, but only ~60% open it. ITT is what shipping the policy
# delivers (diluted); TOT/LATE rescales by compliance — randomized assignment is the instrument.

# %%
assigned = (rng.random(n) < 0.5).astype(int)
opened = assigned * (rng.random(n) < 0.6).astype(int)
order_value = 50.0 + 6.0 * opened + rng.normal(0.0, 8.0, n)

result = causal.itt_tot(assigned, opened, order_value)
print(f"ITT (per assigned)   {result.itt:+.2f}")
print(f"compliance           {result.compliance:.2f}")
print(f"TOT/LATE (per opener) {result.tot:+.2f}   (true +6)")

# %% [markdown]
# ## 4. Confounded discounts — instrumental variables (true effect **-1.5**)
# Reps give bigger discounts exactly when unobserved demand is weak, so OLS is badly biased.
# A supplier cost shock moves the discount but touches sales only *through* it — a valid
# instrument. `iv_effect` = cov(z,y)/cov(z,t).

# %%
cost_shock = rng.normal(0.0, 1.0, n)
weak_demand = rng.normal(0.0, 1.0, n)
discount = 0.8 * cost_shock + weak_demand + rng.normal(0.0, 0.3, n)
sales = -1.5 * discount - 2.0 * weak_demand + rng.normal(0.0, 0.3, n)

ols_slope = float(np.cov(discount, sales)[0, 1] / np.var(discount, ddof=1))
print(f"OLS slope    {ols_slope:+.2f}   (confounded by unobserved demand)")
print(f"IV estimate  {causal.iv_effect(sales, discount, cost_shock):+.2f}   (true -1.5)")

# %% [markdown]
# ## 5. Gold status at 600 points — regression discontinuity (true jump **+15**)
# Customers just below and just above the threshold are exchangeable, so the fitted jump at the
# cutoff is causal *for customers near 600*. Check stability across bandwidths.

# %%
score = rng.uniform(300.0, 900.0, n)
monthly = 0.05 * score + 15.0 * (score >= 600) + rng.normal(0.0, 10.0, n)

for bandwidth in (None, 150.0, 75.0):
    rdd = causal.regression_discontinuity(score, monthly, cutoff=600.0, bandwidth=bandwidth)
    label = "all data" if bandwidth is None else f"±{bandwidth:.0f}"
    print(
        f"bandwidth {label:>8}: effect {rdd.effect:+.1f} "
        f"(se {rdd.std_err:.1f}, p {rdd.p_value:.1g}, n {rdd.n_left}+{rdd.n_right})"
    )

# %% [markdown]
# ## 6. One treated market — synthetic control (true effect **+12**)
# A campaign launches in one region; five untouched regions form the donor pool. Non-negative
# sum-to-one weights build a synthetic twin that tracks the pre-period; the post-period gap is the
# effect. Trust requires a small `pre_rmse`.

# %%
pre_weeks, post_weeks = 40, 16
donors_pre = rng.normal(100.0, 6.0, (pre_weeks, 5)) + np.linspace(0, 8, pre_weeks)[:, None]
donors_post = rng.normal(104.0, 6.0, (post_weeks, 5)) + np.linspace(8, 11, post_weeks)[:, None]
true_mix = np.array([0.5, 0.3, 0.2, 0.0, 0.0])

sc = causal.synthetic_control(
    donors_pre @ true_mix,
    donors_pre,
    donors_post @ true_mix + 12.0,
    donors_post,
    labels=["north", "south", "east", "west", "central"],
)
print(f"effect {sc.effect:+.1f} (true +12)   pre-fit RMSE {sc.pre_rmse:.2f}")
sc.weights

# %%
# Placebo test — the synthetic-control significance check: pretend an *untreated* donor was the
# treated unit and re-run. Its "effect" should hover near zero; if placebos showed effects as
# large as the real one, the +12 would be noise, not signal.
placebo = causal.synthetic_control(
    donors_pre[:, 0],
    donors_pre[:, 1:],
    donors_post[:, 0],
    donors_post[:, 1:],
    labels=["south", "east", "west", "central"],
)
print(
    f"placebo effect on an untreated donor: {placebo.effect:+.1f}   "
    f"(real campaign: {sc.effect:+.1f})"
)

# %% [markdown]
# ## 7. Who responds? Subgroups and uplift modeling (true lift: **+12pp**, engaged only)
# The campaign converts only the engaged half. `subgroup_effects` finds it; a T-learner scores
# *individual* uplift so the next wave targets persuadables — and Qini, not accuracy, judges it.

# %%
engaged = (rng.random(n) < 0.5).astype(int)
treated = (rng.random(n) < 0.5).astype(int)
converted = (rng.random(n) < 0.10 + 0.12 * treated * engaged).astype(int)
campaign = pl.DataFrame(
    {
        "engagement": np.where(engaged == 1, "engaged", "dormant"),
        "treated": treated,
        "converted": converted,
    }
)
causal.subgroup_effects(campaign, outcome="converted", treatment="treated", segment="engagement")

# %%
features = np.column_stack([engaged, rng.normal(0.0, 1.0, n)])  # engagement + a noise feature
learner = causal.TLearner(registry.make_model("logistic", task="classification", max_iter=500)).fit(
    features, treated, converted
)
scores = learner.predict(features)

print(
    f"mean predicted uplift — engaged: {scores[engaged == 1].mean():+.3f}, "
    f"dormant: {scores[engaged == 0].mean():+.3f}"
)
print(f"Qini AUC (model ranking): {causal.qini_auc(converted, treated, scores):+.1f}")
print(f"Qini AUC (random ranking): {causal.qini_auc(converted, treated, rng.normal(size=n)):+.1f}")

# %% [markdown]
# **Takeaways:** each design recovered its planted truth — IPW/matching fixed the voucher bias
# (+10), DiD isolated the rollout (+8), TOT rescaled the diluted ITT (+6), IV undid the discount
# confounding (-1.5), RDD read the loyalty-threshold jump (+15) stably across bandwidths,
# synthetic control measured the single-market launch (+12), and the T-learner pointed the next
# campaign wave at the engaged segment. Pick the estimator by the assumption you can defend.
