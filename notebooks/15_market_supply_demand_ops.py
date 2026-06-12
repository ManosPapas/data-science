# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 15 · Supply & demand — equilibrium, hidden demand, saturation, and capacity economics
#
# Pricing notebooks model one side of the market; this one models the *meeting* of the two.
# Where does price clear the market (and what does a supply shock do to it)? How much demand do
# sold-out periods hide (censored-demand unconstraining — the classic revenue-management step)?
# How much market is left before saturation? How concentrated is it? And the operations half of
# the same questions: how much to stock (newsvendor/EOQ) and how many servers a service level
# really needs (Erlang C) — where utilization economics get violently nonlinear.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Market equilibrium — and what a supply shock does
# Linear demand (q = 500 - 3p) meets linear supply (q = -100 + 5p). The closed form and the
# numeric root agree; then capacity drops 20% (a supplier exits) and the clearing price jumps —
# the equilibrium lens turns "prices are up" from a complaint into an explanation.

# %%
base_eq = pricing.market.linear_equilibrium(
    demand_intercept=500.0, demand_slope=-3.0, supply_intercept=-100.0, supply_slope=5.0
)
numeric = pricing.market.equilibrium(
    lambda p: 500.0 - 3.0 * p, lambda p: -100.0 + 5.0 * p, price_low=20.0, price_high=140.0
)
shocked = pricing.market.equilibrium(
    lambda p: 500.0 - 3.0 * p,
    lambda p: 0.8 * (-100.0 + 5.0 * p),  # 20% supply shock
    price_low=20.0,
    price_high=160.0,
)
print(
    f"equilibrium: €{base_eq.price:.1f} x {base_eq.quantity:.0f} units (numeric agrees: "
    f"€{numeric.price:.1f})"
)
print(f"after the supply shock: €{shocked.price:.1f} x {shocked.quantity:.0f} units")

# %% [markdown]
# ## 2. Market balance over time — where demand and supply mismatch
# Weekly demand vs available capacity for a year: seasonal demand against flat capacity creates
# alternating shortage (lost sales, queues — scarcity you can price into) and surplus (idle
# capacity, markdown pressure). Persistent one-sided gaps are the capacity-planning signal.

# %%
week = np.arange(52)
demand_series = 1000.0 + 250.0 * np.sin(2 * np.pi * week / 52) + rng.normal(0, 40, 52)
capacity_series = np.full(52, 1050.0)
balance = pricing.market.supply_demand_gap(demand_series, capacity_series, labels=week)

print(balance.group_by("regime").agg(pl.len(), pl.col("gap").mean().round(0)))
print(f"unmet demand over the year: {balance['unmet'].sum():,.0f} units")
interactive.line(
    balance.unpivot(on=["demand", "supply"], index="label"),
    "label",
    "value",
    color="variable",
    title="Demand vs capacity — shortage season and surplus season",
)

# %% [markdown]
# ## 3. Unconstraining — the demand the sales data never saw
# When a flight/hotel/SKU sells out, recorded sales = capacity, and true demand is *censored*.
# Averaging raw sales therefore understates demand exactly where it matters most (peak periods).
# The censored-normal MLE recovers the real demand level and prices the spill.

# %%
true_demand = rng.normal(980.0, 120.0, 300)
cap = np.full(300, 1050.0)
sales = np.minimum(true_demand, cap)

unconstrained = pricing.market.unconstrain_demand(sales, cap)
print(f"observed mean sales   {unconstrained.observed_mean:7.0f}")
print(f"estimated true demand {unconstrained.mean:7.0f}  (truth: 980)")
print(
    f"spill {unconstrained.spill:.0f} units/period "
    f"({unconstrained.spill_rate:.1%} of true demand never served, "
    f"{unconstrained.constrained_share:.0%} of periods at capacity)"
)
print("=> forecast and capacity decisions should run on the unconstrained series")

# %% [markdown]
# ## 4. Saturation — how much market is left
# Cumulative adoption follows an S-curve; the logistic fit's ceiling *is* the market potential.
# Before the inflection the ceiling is weakly identified (the curve still looks exponential) —
# treat early-stage capacity estimates as speculative, and date the slowdown with
# `time_to_share`.

# %%
quarters = np.arange(28, dtype=float)
adoption = 80_000.0 / (1.0 + np.exp(-0.35 * (quarters - 14.0))) + rng.normal(0, 900, 28)
s_curve = pricing.market.saturation_fit(quarters, adoption)
print(f"market potential ~{s_curve.capacity:,.0f} customers (r2 {s_curve.r_squared:.3f})")
print(f"current penetration: {adoption[-1] / s_curve.capacity:.0%}")
print(f"90% of potential reached around quarter {s_curve.time_to_share(0.9):.1f}")

fitted = pl.DataFrame(
    {"quarter": quarters, "actual": adoption, "fitted": s_curve.predict(quarters)}
)
interactive.line(
    fitted.unpivot(on=["actual", "fitted"], index="quarter"),
    "quarter",
    "value",
    color="variable",
    title="Adoption S-curve — the ceiling is the market potential",
)

# %% [markdown]
# ## 5. Market structure — share and concentration
# Shares plus the HHI (sum of squared %-shares): the antitrust read is < 1,500 competitive,
# 1,500-2,500 moderate, > 2,500 concentrated. Concentration frames pricing power — both yours
# and the competitor you're about to provoke (notebook 17 picks that thread up with game theory).

# %%
sales_by_player = pl.DataFrame(
    {
        "player": ["us", "rival_a", "rival_b", "long_tail"],
        "revenue": [42_000_000.0, 31_000_000.0, 18_000_000.0, 9_000_000.0],
    }
)
shares = pricing.market.market_share(sales_by_player, value="revenue", by="player")
print(shares)
print(f"HHI: {pricing.market.hhi(shares['share'].to_numpy()):,.0f} -> concentrated market")

# %% [markdown]
# ## 6. Inventory — newsvendor, EOQ, safety stock
# The operations mirror of demand uncertainty. The newsvendor stocks to the *critical fractile*
# (margin vs overstock cost), not to mean demand — fat margins justify deliberate overstock.
# EOQ sizes the order batch; safety stock buys service level through lead time (and its noise).

# %%
stocking = inventory.newsvendor(
    price=10.0, cost=4.0, salvage=1.0, demand_mean=unconstrained.mean, demand_std=unconstrained.std
)
print(
    f"newsvendor: stock {stocking.quantity:.0f} units "
    f"(fractile {stocking.critical_fractile:.0%}, mean demand {unconstrained.mean:.0f}) "
    f"-> expected profit €{stocking.expected_profit:,.0f}"
)

ordering = inventory.eoq(demand=12_000.0, order_cost=150.0, holding_cost=2.5)
print(
    f"EOQ: order {ordering.order_quantity:.0f} at a time, "
    f"{ordering.orders_per_period:.1f} orders/yr, total cost €{ordering.total_cost:,.0f}"
)

buffer = inventory.safety_stock(
    demand_mean=230.0, demand_std=60.0, lead_time=5.0, service_level=0.95, lead_time_std=1.0
)
trigger = inventory.reorder_point(
    demand_mean=230.0, demand_std=60.0, lead_time=5.0, service_level=0.95, lead_time_std=1.0
)
print(f"safety stock {buffer:.0f}, reorder point {trigger:.0f}")

# %%
# Validate the policy against lumpy simulated demand before trusting the closed forms.
sim_demand = rng.normal(230.0, 60.0, 200).clip(min=0)
stock_path = inventory.simulate_inventory_policy(
    sim_demand, reorder_at=trigger, order_quantity=ordering.order_quantity, lead_periods=5
)
print(f"simulated stock-out periods: {(stock_path < 0).sum()} / 200 (target service level 95%)")

# %% [markdown]
# ## 7. Capacity & queueing — why 95% utilization is a trap
# Staffing a support line: 120 calls/hour, 6 minutes average handle time. Erlang C gives the
# wait probability and the SLA number, and `required_servers` inverts it. Watch the wait
# probability explode as utilization approaches 1 — the nonlinearity every "efficiency" target
# forgets.

# %%
metrics = capacity.erlang_c(arrival_rate=120.0, service_rate=10.0, servers=14)
print(
    f"14 agents: utilization {metrics.utilization:.0%}, P(wait) {metrics.wait_probability:.0%}, "
    f"avg wait {metrics.average_wait * 60:.1f} min, "
    f"80%-in-20s SLA: {metrics.service_level(20 / 3600):.0%}"
)

sized = capacity.required_servers(
    arrival_rate=120.0, service_rate=10.0, target_service_level=0.8, answer_within=20 / 3600
)
print(f"to answer 80% within 20s: {sized.servers} agents (utilization {sized.utilization:.0%})")

# %%
sweep = pl.DataFrame(
    [
        {
            "servers": c,
            "utilization": capacity.erlang_c(
                arrival_rate=120.0, service_rate=10.0, servers=c
            ).utilization,
            "wait_probability": capacity.erlang_c(
                arrival_rate=120.0, service_rate=10.0, servers=c
            ).wait_probability,
        }
        for c in range(13, 22)
    ]
)
interactive.line(
    sweep,
    "utilization",
    "wait_probability",
    title="The queueing wall: P(wait) explodes as utilization → 1",
)

# %% [markdown]
# **Takeaways:** the equilibrium frame prices a 20% supply shock at roughly +€9 on the clearing
# price before any competitor reaction; seasonal demand against flat capacity splits the year
# into a shortage season (price/expand) and a surplus season (stimulate/markdown); sold-out
# periods hid ~4% of true demand — forecasts built on raw sales would under-call peak demand
# exactly when it pays most; the S-curve says ~⅔ of the market is already penetrated and growth
# spend should be re-justified against `time_to_share`; the market is HHI-concentrated, so
# pricing moves will be *answered* (notebook 17); the newsvendor deliberately stocks above mean
# demand because the margin structure makes stock-outs dearer than leftovers — and the simulated
# (R, Q) policy confirms the closed forms hold on lumpy demand; and the Erlang sweep shows the
# real capacity trade-off: the last few points of utilization buy exponentially worse waiting,
# so the SLA — not a utilization target — should size the team.
