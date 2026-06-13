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
# (not F1) with the offer values grounded in CLV, stress the campaign business case with
# scenarios and a tornado, price risk itself with expected utility, allocate the budget with a
# linear program and an optimal assignment, and let bandits *learn while deciding*.

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
candidates = np.linspace(50.0, 200.0, 151)
best_price, best_profit = pricing.optimize.optimal_price(
    intercept, elasticity, candidates, unit_cost=unit_cost
)
print(f"grid-search optimum  {best_price:6.1f}  (profit {best_profit:,.0f})")
print(
    f"closed-form markup   {pricing.optimize.markup_price(elasticity, unit_cost):6.1f}  = c·e/(e+1)"
)

# %%
# Revenue and profit want different prices. With elastic demand (e < -1) revenue keeps rising as
# price falls (its "optimum" is the boundary), while profit has a real interior maximum. Optimize
# profit; report revenue.
revenue_curve = pricing.optimize.revenue_at(intercept, elasticity, candidates)
profit_curve_values = pricing.optimize.profit_at(
    intercept, elasticity, candidates, unit_cost=unit_cost
)
print(f"revenue-maximizing candidate: {candidates[int(np.argmax(revenue_curve))]:.0f}  (the floor)")
print(f"profit-maximizing candidate : {candidates[int(np.argmax(profit_curve_values))]:.0f}")

# %%
# The whole trade-off interactively (Plotly — hover for exact values).
price_frame = pl.DataFrame(
    {"price": candidates, "revenue": revenue_curve, "profit": profit_curve_values}
).unpivot(on=["revenue", "profit"], index="price")
interactive.line(
    price_frame, "price", "value", color="variable", title="Revenue vs profit by price"
)

# %% [markdown]
# ## 2. Churn intervention — the threshold that maximizes money, not F1
# Where does the offer's value come from? A quick CLV calculation: at this AOV, frequency, margin
# and churn, a retained customer is worth ~€120 of future margin; the offer costs €10. With costs
# that asymmetric, the profit-maximizing threshold sits far from both 0.5 and the F1 optimum —
# the cost matrix, not the metric, should choose the operating point.

# %%
value_saved = financial.clv(aov=50.0, purchase_frequency=2.4, margin=0.5, churn=0.5)
print(f"CLV of a retained customer ≈ €{value_saved:.0f} -> tp = {value_saved - 10:.0f}, fp = -10")

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


def value_at(threshold: float) -> float:
    return profit.expected_value(y_test, (scores >= threshold).astype(int), costs=offer_costs)


print(f"profit-max threshold {money_threshold:.2f} -> €{money_value:,.0f} on the test set")
print(f"F1-max threshold     {f1_threshold:.2f} -> €{value_at(f1_threshold):,.0f}")
print(f"default 0.5          -> €{value_at(0.5):,.0f}")

# %%
# F1 balances precision and recall symmetrically; the money curve doesn't — a missed churner costs
# 11x a wasted offer, pushing the cut far left.
fig, axes = base.grid(1, ncols=1)
model.threshold_curve(y_test, scores, ax=axes[0], title="Precision / recall / F1 vs threshold")

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

# %%
# Same base case in KPI language: incremental margin vs campaign cost.
incremental = base_case["reach"] * base_case["uplift_pp"] * base_case["margin"]
spend = base_case["reach"] * base_case["contact_cost"]
print(
    f"ROI = {financial.roi(incremental, spend):.0%}  "
    f"(gain €{incremental:,.0f} on €{spend:,.0f} spend)"
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
# ## 5. Allocate the budget — linear programming & optimal assignment
# Decision science's workhorses. The LP spends €60k across channels with different incremental
# margins per euro (and per-channel caps); the Hungarian algorithm pairs account managers to
# regions where each adds the most expected uplift — exactly one each.

# %%
channels = ["email", "social", "search"]
margin_per_euro = [0.9, 1.4, 1.1]
caps = [40_000.0, 25_000.0, 35_000.0]
lp = optimize.linear_program(
    margin_per_euro,
    a_ub=[[1.0, 1.0, 1.0]],
    b_ub=[60_000.0],
    bounds=[(0.0, cap) for cap in caps],
    maximize=True,
)
allocation = pl.DataFrame({"channel": channels, "spend": np.round(lp.x), "cap": caps})
print(f"expected incremental margin: €{-lp.fun:,.0f}")
allocation

# %%
managers = ["Ana", "Ben", "Chloe", "Dmitri"]
regions = ["NA", "EU", "UK", "APAC"]
expected_uplift = [  # k€ if manager (row) takes region (col)
    [12.0, 9.0, 14.0, 8.0],
    [10.0, 11.0, 9.0, 7.0],
    [9.0, 13.0, 11.0, 10.0],
    [8.0, 9.0, 10.0, 12.0],
]
rows_idx, cols_idx = optimize.assign(expected_uplift, maximize=True)
total = sum(expected_uplift[r][c] for r, c in zip(rows_idx, cols_idx, strict=True))
for r, c in zip(rows_idx, cols_idx, strict=True):
    print(f"{managers[r]:6s} -> {regions[c]}  (+€{expected_uplift[r][c]:.0f}k)")
print(f"total expected uplift: €{total:.0f}k")

# %% [markdown]
# ## 6. Learning while deciding — a bandit policy shoot-out
# Three candidate offers convert at (unknown to the policies) 4% / 8% / 5%. Each policy plays
# 4,000 rounds on an identical reward stream: Thompson sampling samples from Beta posteriors,
# epsilon-greedy explores at a fixed 10% tax, UCB1 follows optimism bounds. The always-best
# oracle expects 320 conversions — a fixed equal-split A/B only 227.

# %%
true_rates = [0.04, 0.08, 0.05]
policies = {
    "thompson": bandits.ThompsonSampling(3, seed=42),
    "epsilon_greedy": bandits.EpsilonGreedy(3, epsilon=0.1, seed=42),
    "ucb1": bandits.UCB1(3),
}
for name, policy in policies.items():
    stream = np.random.default_rng(7)  # identical luck for every policy
    conversions = 0.0
    for _ in range(4000):
        arm = policy.select()
        reward = float(stream.random() < true_rates[arm])
        policy.update(arm, reward)
        conversions += reward
    print(f"{name:14s} {conversions:.0f} conversions")

# %%
# Inside the winner: Thompson's per-arm Beta posteriors — uncertainty that acts.
thompson = policies["thompson"]
posterior_means = thompson.alpha / (thompson.alpha + thompson.beta)
pulls = (thompson.alpha + thompson.beta - 2.0).astype(int)
for arm, (mean, n) in enumerate(zip(posterior_means, pulls, strict=True)):
    print(f"offer {arm}: true {true_rates[arm]:.0%}, posterior {mean:.3f}, pulled {n} times")

# %% [markdown]
# **Takeaways:** demand is elastic (-1.8): revenue wants the price floor but profit peaks at ~€90,
# where both optimizers agree; the €-optimal churn threshold (grounded in a €120 CLV) treats far
# more customers than the F1 threshold because a missed churner costs 11x a wasted offer; the
# tornado says the uplift assumption — not cost or reach — decides the campaign (validate it with
# notebooks 04/09/10), and the base case clears a healthy ROI; the safe channel beats the
# higher-EV moonshot for a risk-averse budget; the LP and the Hungarian assignment turn those
# value estimates into an actual plan; and Thompson sampling routes ~85% of traffic to the
# winning offer — while UCB1's optimism pays a heavy exploration bill on closely spaced arms,
# barely beating a fixed equal split.
