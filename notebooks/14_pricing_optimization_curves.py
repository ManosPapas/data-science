# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 14 · Price & revenue optimization — marginal economics, curve calculus, dynamic pricing
#
# The revenue function is R(p) = p · D(p): everything in this notebook is calculus on that curve
# and its profit sibling. We read optima three ways (closed form, grid, numerical derivatives),
# verify the curve's shape *before* trusting an interior optimum, plan a price **path** for fixed
# inventory over a deadline (revenue-management DP), and close with the diagnostic side of
# pricing: where realized revenue leaks, and a price/volume/mix bridge of the year's change.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Marginal revenue & marginal profit — the adjustment signal
# Constant-elasticity demand with e = -1.8, unit cost €40. MR = p(1 + 1/e) is below price
# everywhere (selling one more unit means a lower price on *all* units) and crosses cost exactly
# at the profit optimum — so the *sign* of marginal profit says raise vs cut even when the level
# of the demand fit is rough.

# %%
price = rng.uniform(60.0, 140.0, 400)
quantity = np.exp(8.0) * price**-1.8 * rng.lognormal(0.0, 0.15, 400)
intercept, e = pricing.elasticity.fit_demand(price, quantity)
unit_cost = 40.0

closed_form = pricing.optimize.markup_price(e, unit_cost)
grid = np.linspace(45.0, 180.0, 271)
grid_price, grid_profit = pricing.optimize.optimal_price(intercept, e, grid, unit_cost=unit_cost)
print(f"closed-form optimum €{closed_form:.1f}   grid optimum €{grid_price:.1f}")

for p_check in (70.0, closed_form, 130.0):
    mp = pricing.optimize.marginal_profit(e, np.array([p_check]), unit_cost=unit_cost)[0]
    direction = "raise" if mp < 0 else ("cut" if mp > 0 else "hold")
    print(f"  marginal profit at €{p_check:6.1f}: {mp:7.2f}  -> {direction}")

# %% [markdown]
# ## 2. The curve itself — derivatives, turning points, convexity
# `analytics.curves` is the generic calculus layer: the same machinery works on any sampled
# business curve (spend→conversions, capacity→throughput). Here we confirm the profit curve's
# peak numerically and — more important — check **concavity**, which is what licenses trusting an
# interior optimum from a local search at all.

# %%
profit_curve = pricing.optimize.profit_at(intercept, e, grid, unit_cost=unit_cost)
peak = curves.local_extrema(grid, profit_curve)
shape = curves.convexity(grid, profit_curve)
print(peak)
print(f"curve verdict: {shape.verdict} (concave share {shape.concave_share:.0%})")
print("concave around the peak -> the interior optimum is trustworthy")

# %%
# Marginal revenue read straight off the sampled revenue curve: slope crosses zero at the
# revenue (not profit) peak — and point elasticity recovers e at every price.
revenue_curve = pricing.optimize.revenue_at(intercept, e, grid)
marginal_from_curve = curves.slope(grid, revenue_curve)
local_e = curves.point_elasticity(grid, revenue_curve)
print(f"point elasticity of revenue curve: {local_e[10]:.2f} (= 1 + e = {1 + e:.2f})")
frame = pl.DataFrame(
    {"price": grid, "revenue": revenue_curve, "profit": profit_curve, "mr": marginal_from_curve}
)
decision.price_curves(
    frame,
    curves=("revenue", "profit"),
    optimum=grid_price,
    title="Revenue vs profit — elastic demand wants different prices for each",
)

# %% [markdown]
# ## 3. Linear demand — the other closed form
# Where log-log demand never chokes off, a linear curve hits zero at the choke price and the
# profit-optimal price is exactly midway between unit cost and choke. Useful as a *local*
# approximation around the current price even when the global curve is curved.

# %%
linear = pricing.demand.fit_linear_demand(price, quantity)
best_linear = pricing.optimize.optimal_price_linear(
    linear.intercept, linear.slope, unit_cost=unit_cost
)
print(
    f"choke €{linear.choke_price:.0f}, linear-model optimum €{best_linear:.0f} "
    f"(midpoint of cost {unit_cost:.0f} and choke)"
)
print(f"local elasticity at the optimum: {linear.elasticity_at([best_linear])[0]:.2f}")

# %% [markdown]
# ## 4. Dynamic pricing — a fixed stock and a deadline
# 40 seats/rooms/units, 10 selling periods, Poisson demand that falls with price. Backward
# induction solves the whole policy: the optimal price for *every* (time-left, stock-left) state.
# The two classic revenue-management forces appear by themselves: prices fall as the deadline
# nears (perishable inventory) and rise when stock runs scarce.


# %%
def demand_rate(p: float, t: int) -> float:
    return 9.0 * (p / 100.0) ** -2.2  # elastic demand, stationary within the horizon


policy = pricing.optimize.dynamic_prices(
    demand_rate, capacity=40, periods=10, candidates=np.arange(70.0, 151.0, 10.0)
)
print(f"expected revenue under the optimal policy: €{policy.expected_revenue:,.0f}")
print(f"price tonight with 40 left: €{policy.price_for(0, 40):.0f}")
print(f"price tonight with  5 left: €{policy.price_for(0, 5):.0f}   (scarcity premium)")
print(f"price in period 9 with 40 left: €{policy.price_for(9, 40):.0f} (deadline discount)")

# %%
states = policy.policy_frame().filter(pl.col("remaining").is_in([5, 20, 40]))
interactive.line(
    states.with_columns(pl.col("remaining").cast(pl.String)),
    "period",
    "price",
    color="remaining",
    title="Optimal price path by remaining inventory",
)

# %% [markdown]
# ## 5. Revenue leakage — entitled vs realized price
# Optimization sets the price; execution leaks it. Compare contract/list revenue with what
# actually landed, grouped by sales rep: systematic gaps are unapproved discounting, fee waivers,
# or billing misses — margin recovered here is pure profit, no demand risk attached.

# %%
reps = rng.choice(["ana", "ben", "chloe", "dmitri"], 2000)
list_price = rng.uniform(80.0, 120.0, 2000)
discount = rng.uniform(0.0, 0.05, 2000) + np.where(reps == "ben", 0.12, 0.0)  # ben over-discounts
deals = pl.DataFrame(
    {"rep": reps, "list_revenue": list_price, "realized": list_price * (1.0 - discount)}
)
drivers.revenue_leakage(deals, expected="list_revenue", actual="realized", by="rep")

# %% [markdown]
# ## 6. The price/volume/mix bridge — why revenue moved
# Finance asks "price or volume?"; the honest answer usually includes *mix*. The bridge splits
# ΔRevenue into the three effects exactly (no residual): price (charging differently), volume
# (the market growing), mix (share shifting between cheap and expensive segments) — separating
# "our pricing worked" from "the mix flattered us".

# %%
last_year = pl.DataFrame(
    {
        "product": ["economy", "flex", "premium"],
        "price": [80.0, 130.0, 220.0],
        "volume": [6000.0, 2500.0, 800.0],
    }
)
this_year = pl.DataFrame(
    {
        "product": ["economy", "flex", "premium"],
        "price": [84.0, 132.0, 215.0],  # economy repriced +5%
        "volume": [5400.0, 2900.0, 1150.0],  # premium mix grew
    }
)
bridge = drivers.price_volume_mix(
    this_year, last_year, price="price", volume="volume", by="product"
)
bridge

# %%
totals = bridge.select(["price_effect", "volume_effect", "mix_effect", "total_effect"]).sum()
print(totals)
print("volume slightly down, but price and premium-mix more than compensate")

# %% [markdown]
# **Takeaways:** closed form, grid search, and the numerical derivative all land on the same
# profit-maximizing price (~€90), and the marginal-profit *sign* gives the raise/cut direction
# even when the fit is rough; the profit curve is concave around its peak, which is what makes
# the interior optimum trustworthy (always check before believing a solver); the dynamic-pricing
# DP prices scarcity up (+€40 when stock is short) and deadlines down — one policy table covers
# every state; Ben's desk leaks ~12% of entitled revenue, which is margin recoverable without
# touching demand; and the year's revenue growth decomposes into +price +mix against -volume, a
# very different story (and action list) than the topline alone suggests.
