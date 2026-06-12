# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 12 · Decision science & pricing — from estimates to money
#
# Models end in decisions. Here the last mile: estimate price elasticity and set the
# revenue/profit-maximizing price, pick the churn-intervention threshold that maximizes money
# (not F1), stress the campaign business case with scenarios and a tornado, price risk itself
# with expected utility, and let a bandit *learn while deciding* which price to offer.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

customers = read_parquet(ROOT / "data" / "raw" / "customers.parquet").pipe(
    clean.fill_missing, strategy="median", columns=["monthly_spend", "satisfaction"]
)

# %% [markdown]
# ## 1. Price elasticity → optimal price
# 26 weeks of A/B-tested price points (randomized, so the log-log fit is causal — with
# observational prices you'd reach for the IV machinery of notebook 09; true elasticity **-1.8**).
# The closed-form markup optimum and the grid search should agree.

# %%
price = rng.uniform(60.0, 140.0, 260)
quantity = np.exp(8.0) * price**-1.8 * rng.lognormal(0.0, 0.15, 260)
intercept, elasticity = pricing.elasticity.fit_demand(price, quantity)
print(f"elasticity {elasticity:.2f} (true -1.8) — elastic, so price cuts grow revenue")

unit_cost = 40.0
best_price, best_profit = pricing.optimize.optimal_price(
    intercept, elasticity, candidates=np.linspace(50.0, 200.0, 151), unit_cost=unit_cost
)
print(f"grid-search optimum  {best_price:6.1f}  (profit {best_profit:,.0f})")
print(
    f"closed-form markup   {pricing.optimize.markup_price(elasticity, unit_cost):6.1f}  = c·e/(e+1)"
)

# %% [markdown]
# ## 2. Churn intervention — the threshold that maximizes money, not F1
# A €10 retention offer saves ~€120 of margin when it lands on a genuine churner. With costs that
# asymmetric, the profit-maximizing threshold sits far from both 0.5 and the F1 optimum — the
# cost matrix, not the metric, should choose the operating point.

# %%
drivers = [
    "age",
    "tenure_months",
    "num_products",
    "sessions_30d",
    "support_tickets",
    "monthly_spend",
    "satisfaction",
]
train_df, test_df = split.train_test_split(customers, test_size=0.3, stratify="churned", seed=42)
churn_model = train.fit(
    registry.make_model("logistic", task="classification", max_iter=2000),
    train_df.select(drivers),
    train_df["churned"],
)
scores = train.predict_proba(churn_model, test_df.select(drivers))[:, 1]
y_test = test_df["churned"].to_numpy()

offer_costs = {"tp": 110.0, "fp": -10.0}  # saved margin net of offer; wasted offer
money_threshold, money_value = profit.profit_threshold(y_test, scores, costs=offer_costs)
f1_threshold = imbalance.tune_threshold(y_test, scores, metric="f1")

value_at = lambda t: profit.expected_value(y_test, (scores >= t).astype(int), costs=offer_costs)  # noqa: E731
print(f"profit-max threshold {money_threshold:.2f} -> €{money_value:,.0f} on the test set")
print(f"F1-max threshold     {f1_threshold:.2f} -> €{value_at(f1_threshold):,.0f}")
print(f"default 0.5          -> €{value_at(0.5):,.0f}")

# %% [markdown]
# ## 3. The campaign business case — scenarios, then the tornado
# Value = reach * uplift * margin - reach * cost. Scenarios stress coherent futures; the
# sensitivity table ranks which single assumption the decision actually hinges on — spend the
# next research euro there.


# %%
def campaign_value(reach: float, uplift_pp: float, margin: float, contact_cost: float) -> float:
    return reach * uplift_pp * margin - reach * contact_cost


base_case = {"reach": 50_000.0, "uplift_pp": 0.03, "margin": 120.0, "contact_cost": 1.5}
scenario.scenario_table(
    campaign_value,
    base_case,
    {
        "downturn": {"uplift_pp": 0.02, "margin": 100.0},
        "optimistic": {"uplift_pp": 0.045},
        "costs blow up": {"contact_cost": 3.0},
    },
)

# %%
scenario.sensitivity(
    campaign_value,
    base_case,
    {
        "uplift_pp": (0.015, 0.045),
        "margin": (90.0, 150.0),
        "contact_cost": (1.0, 3.0),
        "reach": (30_000.0, 70_000.0),
    },
)

# %% [markdown]
# ## 4. Risk itself has a price — expected utility
# Two ways to spend the budget: a safe channel returning €40k for sure, or a moonshot returning
# €120k with 40% probability. The moonshot has the *higher* expected value (€48k vs €40k), yet a
# moderately risk-averse decision-maker still takes the sure thing: the certainty equivalent
# prices the gamble in money, and EV - CE is the risk premium.

# %%
safe = scenario.certainty_equivalent([40.0], [1.0], risk_aversion=0.02)  # k€ units
moonshot_ev = scenario.expected_utility([0.0, 120.0], [0.6, 0.4])
moonshot_ce = scenario.certainty_equivalent([0.0, 120.0], [0.6, 0.4], risk_aversion=0.02)
print(f"moonshot expected value      €{moonshot_ev:5.1f}k")
print(
    f"moonshot certainty equivalent €{moonshot_ce:5.1f}k at a=0.02 "
    f"(risk premium €{moonshot_ev - moonshot_ce:.1f}k)"
)
print(f"safe option                   €{safe:5.1f}k -> the risk-averse buyer takes the sure €40k")

# %% [markdown]
# ## 5. Learning while deciding — Thompson sampling over price points
# Three candidate offers convert at (unknown to the bandit) 4% / 8% / 5%. Thompson sampling keeps
# a Beta posterior per arm and samples from them to choose — exploration self-tunes away as the
# evidence builds, unlike a fixed-split A/B that keeps paying for the losers.

# %%
true_rates = [0.04, 0.08, 0.05]
bandit = bandits.ThompsonSampling(3, seed=42)
for _ in range(4000):
    arm = bandit.select()
    bandit.update(arm, float(rng.random() < true_rates[arm]))

posterior_means = bandit.alpha / (bandit.alpha + bandit.beta)
pulls = (bandit.alpha + bandit.beta - 2.0).astype(int)
for arm, (mean, n) in enumerate(zip(posterior_means, pulls, strict=True)):
    print(f"offer {arm}: true {true_rates[arm]:.0%}, posterior {mean:.3f}, pulled {n} times")

# %% [markdown]
# **Takeaways:** demand is elastic (-1.8) and both optimizers agree the profit price is ~€90;
# the €-optimal churn threshold treats far more customers than the F1 threshold because a missed
# churner costs 11x a wasted offer; the tornado says the uplift assumption — not cost or reach —
# decides the campaign, so validate it first (notebooks 04/09/10 are exactly that toolkit); the
# safe channel beats the higher-EV moonshot for a risk-averse budget; and the bandit concentrated
# its traffic on the winning offer while still keeping tabs on the others.
