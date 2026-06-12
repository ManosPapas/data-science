# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 10 · Bayesian methods — beliefs, updated by data, read as decisions
#
# The Bayesian grammar (prior · likelihood → posterior) on real questions: a fraud alert worth
# triaging (Bayes' rule), a landing-page rate with thin data (conjugate Beta posterior), the
# retention A/B read as P(better) and expected loss instead of a p-value, a churn league table
# fixed by hierarchical shrinkage, and an MCMC sampler for when no closed form exists.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()

customers = read_parquet(ROOT / "data" / "raw" / "customers.parquet")

# %% [markdown]
# ## 1. Bayes' rule — the base-rate guard
# A fraud model fires on a transaction: 99% sensitive, 5% false-positive rate, but only 1% of
# transactions are fraud. The posterior is ~17%, not 99% — evidence *updates* the prior, it never
# replaces it. This arithmetic is the whole module in miniature.

# %%
posterior = stats.bayes_rule(prior=0.01, true_positive_rate=0.99, false_positive_rate=0.05)
print(f"P(fraud | alert) = {posterior:.1%}   (despite a 99%-sensitive model)")

# %% [markdown]
# ## 2. One rate, thin data — conjugate Beta posterior
# A new landing page: 18 conversions in 120 visits. With a uniform prior the interval is wide;
# encoding history (last quarter ran ~20%) tightens it. The interval is *credible*: "the rate is
# in here with 95% probability" — the reading a frequentist CI doesn't license.

# %%
flat = bayes.beta_posterior(18, 120)
informed = bayes.beta_posterior(18, 120, prior=(8.0, 32.0))  # history: ~20% on 40 obs
print(f"uniform prior : {flat.mean:.3f}  [{flat.lower:.3f}, {flat.upper:.3f}]")
print(f"informed prior: {informed.mean:.3f}  [{informed.lower:.3f}, {informed.upper:.3f}]")

# %%
# Side by side with the frequentist Wilson interval on the same 18/120: numerically close here,
# but only the credible interval licenses "the rate is in this range with 95% probability" —
# and only the Bayesian version has a dial for prior knowledge.
rate, wilson_lo, wilson_hi = stats.proportion_confidence_interval(18, 120)
print(f"Wilson (frequentist): {rate:.3f}  [{wilson_lo:.3f}, {wilson_hi:.3f}]")

# %%
# Prior sensitivity — the report you attach when priors are debatable. With this much data the
# posterior barely moves across a skeptic, a uniform, and an optimist: the data dominate.
for label, prior in [
    ("skeptic 10%", (4.0, 36.0)),
    ("uniform", (1.0, 1.0)),
    ("optimist 25%", (10.0, 30.0)),
]:
    post = bayes.beta_posterior(18, 120, prior=prior)
    print(f"{label:12s} -> {post.mean:.3f}  [{post.lower:.3f}, {post.upper:.3f}]")

# %% [markdown]
# ## 3. The retention A/B, two ways — p-value vs decision quantities
# Same data as notebook 04. The frequentist read lands exactly on the alpha = 0.05 fence and
# returns "inconclusive" — a coin-flip verdict. The Bayesian read answers the shipping question
# directly: P(treatment better) ≈ 97% and the expected loss if we ship and are wrong is
# negligible — ship. Same data, a decision instead of a fence.

# %%
ctrl = customers.filter(pl.col("group") == "control")
trt = customers.filter(pl.col("group") == "treatment")
ctrl_retained = int((ctrl["churned"] == 0).sum())
trt_retained = int((trt["churned"] == 0).sum())

print(experiment.analyze_conversions(ctrl_retained, ctrl.height, trt_retained, trt.height))

# %%
bayesian = experiment.bayes_conversions(ctrl_retained, ctrl.height, trt_retained, trt.height)
print(f"P(treatment better)   {bayesian.prob_treatment_better:.1%}")
print(f"expected loss if shipped {bayesian.expected_loss:.5f} (retention points at risk)")
print(f"95% credible interval for the lift {bayesian.credible_interval}")

# %%
# Guardrail (spend) the same way — the posterior of each mean via the CLT-normal approximation.
spend_ctrl = ctrl["monthly_spend"].drop_nulls().to_numpy()
spend_trt = trt["monthly_spend"].drop_nulls().to_numpy()
guardrail = experiment.bayes_means(spend_ctrl, spend_trt)
print(
    f"P(treatment spends more) {guardrail.prob_treatment_better:.1%} — no movement, guardrail holds"
)

# %% [markdown]
# ## 4. Hierarchical shrinkage — fixing the churn league table
# Churn by region · segment: tiny cells (wealth in any region is ~5% of the base) top and bottom
# the raw ranking by luck. `hierarchical_rates` fits one shared Beta prior and shrinks each cell
# toward the pool in proportion to its evidence — rank on `shrunk_rate`.

# %%
cells = (
    customers.group_by("region", "segment")
    .agg(pl.col("churned").sum().alias("churned"), pl.len().alias("n"))
    .with_columns((pl.col("region") + " · " + pl.col("segment")).alias("cell"))
    .sort("cell")
)
league, prior = bayes.hierarchical_rates(
    cells["churned"].to_numpy(), cells["n"].to_numpy(), labels=cells["cell"].to_list()
)
print(f"fitted shared prior: Beta({prior[0]:.1f}, {prior[1]:.1f})")
league.with_columns((pl.col("rate") - pl.col("shrunk_rate")).abs().alias("moved")).sort(
    "moved", descending=True
).head(6)

# %%
# Raw vs shrunk ranking — the small-n cells move; the big retail cells barely budge.
print("worst by raw rate   :", league.sort("rate", descending=True)["group"].head(3).to_list())
print(
    "worst by shrunk rate:", league.sort("shrunk_rate", descending=True)["group"].head(3).to_list()
)

# %%
# The shrinkage picture: every cell's raw rate vs its shrunk rate. Points off the diagonal are
# being pulled toward the pooled mean — and the further out a raw rate sits, the harder it's
# pulled (those are exactly the thin cells).
fig, axes = base.grid(1, ncols=1)
eda.scatter(league, "rate", "shrunk_rate")

# %% [markdown]
# ## 5. No closed form? MCMC
# Posterior for the mean of log-spend under a normal likelihood with a flat prior — deliberately
# simple so we can check the sampler against the t-interval. Tune `step` toward ~20-40%
# acceptance; for many-parameter models use a dedicated PPL.

# %%
log_spend = np.log(customers["monthly_spend"].drop_nulls().to_numpy())
scale = float(log_spend.std(ddof=1))


def log_density(theta: np.ndarray) -> float:
    return float(-0.5 * np.sum((log_spend - theta[0]) ** 2) / scale**2)


samples, acceptance = bayes.mcmc_sample(
    log_density, start=[float(log_spend.mean())], n_samples=4000, burn_in=500, step=0.02, seed=42
)
mcmc_lo, mcmc_hi = np.quantile(samples[:, 0], [0.025, 0.975])
t_mean, t_lo, t_hi = stats.mean_confidence_interval(log_spend)
print(f"acceptance rate {acceptance:.0%}")
print(f"MCMC posterior  {samples[:, 0].mean():.4f}  [{mcmc_lo:.4f}, {mcmc_hi:.4f}]")
print(f"t interval      {t_mean:.4f}  [{t_lo:.4f}, {t_hi:.4f}]   (they should agree)")

# %%
# The posterior is a *distribution*, not a point — and the draws are ordinary data: histogram
# them, quantile them, feed them into any downstream decision calculation.
fig, axes = base.grid(1, ncols=1)
eda.histogram(samples[:, 0], ax=axes[0], title="Posterior of mean log-spend (MCMC draws)")

# %% [markdown]
# **Takeaways:** the alert posterior (~17%) shows why base rates rule triage; thin-data rates get
# honest intervals from conjugate updates, and history enters explicitly as the prior; the
# retention A/B ships on P(better) ≈ high with negligible expected loss while the spend guardrail
# stays flat; shrinkage reorders the churn league table away from tiny-cell noise; and the MCMC
# interval matches the analytic one — the sampler is trustworthy where no formula exists. For
# online decisions, `bandits.ThompsonSampling` (notebook 12) is this posterior logic acting.
