# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 20 · Survival analysis — *when* customers churn, with censoring done right
#
# Notebook 03 predicted *whether* a customer churns; this one models **when** — and fixes the
# subtle bug in every naive churn rate: customers who haven't churned *yet* are not negatives,
# they are **censored** (we only know they survived this long). Kaplan-Meier uses that exposure
# correctly; Cox regression turns covariates into hazard ratios; restricted mean survival prices
# retention time straight into CLV. Synthetic contracts with a known churn process, so every
# estimate has a truth to hit.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Contracts, churn times, and the censoring problem
# 4,000 customers signed up over the past 36 months. True lifetimes are exponential-ish, shorter
# for customers with many support tickets and longer on the annual plan. We only *observe* a
# churn if it happened before today — everyone else is censored at their current tenure.

# %%
n = 4000
tickets = rng.poisson(1.5, n).astype(float)
annual_plan = (rng.random(n) < 0.4).astype(float)
true_scale = 14.0 * np.exp(-0.28 * tickets + 0.55 * annual_plan)
true_lifetime = rng.exponential(true_scale)

signup_age = rng.uniform(1.0, 36.0, n)  # months since each signup
tenure = np.minimum(true_lifetime, signup_age)
churned = (true_lifetime <= signup_age).astype(float)
customers = pl.DataFrame(
    {
        "tenure": tenure,
        "churned": churned,
        "tickets": tickets,
        "annual_plan": annual_plan,
    }
)
print(f"observed churners: {churned.mean():.0%}; the rest are censored, not 'safe'")

# %% [markdown]
# ## 2. Why the naive rate misleads — and the KM curve doesn't
# "Average tenure of churned customers" drops everyone still alive; the Kaplan-Meier estimator
# keeps every account in the denominator for exactly as long as it was observed. The naive
# number *understates* customer lifetime badly.

# %%
naive_mean_tenure = customers.filter(pl.col("churned") == 1)["tenure"].mean()
km = survival.kaplan_meier(customers["tenure"], customers["churned"])
median_life = survival.median_survival(customers["tenure"], customers["churned"])
print(f"naive mean tenure of churned customers: {naive_mean_tenure:.1f} months")
print(f"KM median customer lifetime:            {median_life:.1f} months")

timeseries.survival_curve(km, title="Kaplan-Meier survival curve with 95% CI")

# %% [markdown]
# ## 3. Retention curves by segment
# The plan question, answered correctly: KM per segment. Annual-plan customers survive visibly
# longer at every tenure — and because censoring is handled, the comparison is fair even though
# the annual plan launched later (its customers have shorter observation windows).

# %%
curves_by_plan = {}
for label, segment in [("monthly", 0.0), ("annual", 1.0)]:
    part = customers.filter(pl.col("annual_plan") == segment)
    curves_by_plan[label] = survival.kaplan_meier(part["tenure"], part["churned"])
timeseries.survival_curve(
    curves_by_plan, title="Retention by plan — annual contracts survive longer at every tenure"
)

# %%
for label, segment in [("monthly", 0.0), ("annual", 1.0)]:
    part = customers.filter(pl.col("annual_plan") == segment)
    at_12 = survival.survival_at(part["tenure"], part["churned"], [12.0])[0]
    print(f"{label:8s} 12-month retention: {at_12:.0%}")

# %% [markdown]
# ## 4. What drives the hazard — Cox proportional hazards
# Hazard ratios with CIs: HR = 1.3 on tickets means each extra ticket raises the churn hazard
# 30% *at every tenure* (that's the proportional-hazards assumption). True effects are
# exp(0.28) ≈ 1.32 per ticket and exp(-0.55) ≈ 0.58 for the annual plan — check the recovery.

# %%
survival.cox_ph(customers, duration="tenure", event="churned", x=["tickets", "annual_plan"])

# %% [markdown]
# ## 5. From survival to money — RMST and CLV
# Restricted mean survival = expected retained months over a horizon (the area under KM), which
# is the number CLV actually needs — always estimable under censoring, unlike the unrestricted
# mean. At €18 margin/month, the annual-plan retention edge prices itself.

# %%
margin_per_month = 18.0
for label, segment in [("monthly", 0.0), ("annual", 1.0)]:
    part = customers.filter(pl.col("annual_plan") == segment)
    expected_months = survival.restricted_mean_survival(
        part["tenure"], part["churned"], horizon=24.0
    )
    print(
        f"{label:8s} expected retained months (24m horizon): {expected_months:5.1f} "
        f"-> 24m CLV ≈ €{expected_months * margin_per_month:,.0f}"
    )

# %% [markdown]
# ## 6. Targeting the intervention
# Cox says tickets drive hazard; the retention play is to fix the *high-ticket* group. Size the
# prize: move a 4-ticket customer's expected retained months to the 1-ticket curve and multiply
# by margin — that per-customer value is the budget ceiling for the support investment
# (cf. notebook 12's profit-threshold framing).

# %%
prize_rows = []
for ticket_level in (1.0, 4.0):
    part = customers.filter(pl.col("tickets") == ticket_level)
    if part["churned"].sum() == 0:
        continue
    months = survival.restricted_mean_survival(part["tenure"], part["churned"], horizon=24.0)
    prize_rows.append({"tickets": ticket_level, "expected_months_24m": months})
prize = pl.DataFrame(prize_rows)
print(prize)
gap = prize["expected_months_24m"][0] - prize["expected_months_24m"][1]
print(f"closing the ticket gap is worth ≈ €{gap * margin_per_month:,.0f} per customer (24m)")

# %% [markdown]
# **Takeaways:** the naive "mean tenure of churners" reads ~9 months while the censoring-correct
# median lifetime is meaningfully longer — survivors count, and ignoring them biases every
# retention number pessimistic; annual-plan customers retain better at every tenure and the Cox
# model prices the drivers cleanly (HR ≈ 1.3 per support ticket, ≈ 0.6 for the annual plan —
# both matching the data-generating truth); restricted mean survival converts curves into the
# CLV currency (expected retained months x margin), putting a defensible euro value on the
# annual-plan push and a per-customer budget ceiling on fixing the high-ticket experience. Next
# step when ticking-clock features arrive (usage decay, price changes): re-fit Cox on tenure
# splits to verify proportional hazards before trusting one HR across the lifecycle.
