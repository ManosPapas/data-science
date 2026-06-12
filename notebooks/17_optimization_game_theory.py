# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 17 · The optimization toolkit — LP duals to integer programs, portfolios, and game theory
#
# Notebook 12 ran one LP and one assignment. This is the full decision-engineering kit: shadow
# prices (what a constraint *costs*), integer programs (where rounding lies), nonlinear budget
# allocation under diminishing returns, mean-variance portfolios, Pareto frontiers when no single
# objective is honest, optimization under uncertainty (mean vs worst-case), network optimization
# (bottlenecks, cheapest backbones) — and game theory, because the competitor's optimizer runs
# too.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. The LP, plus the question finance actually asks
# Allocate €60k across three channels (margins 0.9/1.4/1.1 per euro, per-channel caps). The
# *solution* says where money goes; the **shadow prices** say what one more euro of each
# constraint is worth — the rational ceiling on paying for extra capacity, and the principled
# opportunity-cost number.

# %%
channels = ["email", "social", "search"]
margin_per_euro = [0.9, 1.4, 1.1]
caps = [40_000.0, 25_000.0, 35_000.0]
lp = optimize.linear_program(
    margin_per_euro,
    a_ub=[[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],  # total budget; social's own cap as a row
    b_ub=[60_000.0, 25_000.0],
    bounds=[(0.0, cap) for cap in caps],
    maximize=True,
)
print(pl.DataFrame({"channel": channels, "spend": np.round(lp.x)}))
optimize.shadow_prices(lp, names_ub=["total budget", "social cap"], maximize=True)

# %% [markdown]
# The budget's shadow price (1.1) is the *marginal* channel's rate — not the best channel's:
# extra money flows to search once social is capped. Social's cap is itself worth 0.3/€ (1.4
# minus the 1.1 the euro would otherwise earn) — that's the business case for raising it.

# %% [markdown]
# ## 2. Integer decisions — where LP-and-round lies
# Picking projects under a €50k budget is a knapsack: all-or-nothing. The LP relaxation funds
# 60% of a project, which is meaningless; rounding it off picks the wrong set. The MILP solves
# the actual problem exactly.

# %%
projects = pl.DataFrame(
    {
        "project": ["crm_migration", "pricing_engine", "churn_program", "self_service"],
        "value": [60.0, 100.0, 120.0, 95.0],  # k€ NPV
        "cost": [10.0, 20.0, 30.0, 25.0],  # k€ investment
    }
)
chosen = optimize.knapsack(projects["value"], projects["cost"], capacity=50.0)
print(projects.with_row_index().filter(pl.col("index").is_in(chosen.chosen)))
print(f"portfolio value €{chosen.total_value:.0f}k for €{chosen.total_weight:.0f}k spend")

# Greedy by value-per-euro takes crm (ratio 6) then pricing (5); nothing else fits the
# remaining €20k. Greedy: €160k. The exact MILP instead pairs pricing + churn for €220k.
greedy_order = projects.with_columns((pl.col("value") / pl.col("cost")).alias("ratio")).sort(
    "ratio", descending=True
)
print(f"greedy value-per-euro order: {greedy_order['project'].to_list()}")
print(f"greedy portfolio value €160k vs exact €{chosen.total_value:.0f}k — rounding LPs lies")

# %% [markdown]
# ## 3. Diminishing returns — nonlinear allocation
# Real channel response is concave (√spend here): the LP's "all-in on the best channel" answer
# is wrong once returns diminish. The nonlinear program equalizes *marginal* returns instead —
# textbook microeconomics, solved numerically with a budget constraint.

# %%
effectiveness = np.array([90.0, 140.0, 110.0])


def campaign_value(spend: np.ndarray) -> float:
    return float(np.sum(effectiveness * np.sqrt(spend)))


allocation = optimize.nonlinear_program(
    campaign_value,
    x0=[20_000.0, 20_000.0, 20_000.0],
    bounds=[(0.0, 60_000.0)] * 3,
    constraints=[{"type": "eq", "fun": lambda s: float(s.sum() - 60_000.0)}],
    maximize=True,
)
nonlinear_spend = pl.DataFrame({"channel": channels, "spend": np.round(allocation.x)})
print(nonlinear_spend)
print("no corner solution: every channel gets budget, proportional to effectiveness²")

# %% [markdown]
# ## 4. Portfolio thinking — diversification is the covariance term
# Spread the budget over three product bets with estimated returns and a covariance (two bets
# are correlated — they share a market). Mean-variance weights penalize the correlated pair:
# diversification comes from covariance, not from counting bets.

# %%
expected_returns = np.array([0.12, 0.10, 0.11])
covariance = np.array(
    [
        [0.040, 0.030, 0.002],  # bets 1 & 2 highly correlated
        [0.030, 0.045, 0.002],
        [0.002, 0.002, 0.030],
    ]
)
portfolio = optimize.portfolio_weights(expected_returns, covariance, risk_aversion=3.0)
print(pl.DataFrame({"bet": ["alpha", "beta", "gamma"], "weight": portfolio.weights.round(3)}))
print(
    f"expected return {portfolio.expected_return:.1%}, "
    f"volatility {portfolio.volatility:.1%} — gamma is over-weighted vs its raw return "
    "because it diversifies"
)

# %% [markdown]
# ## 5. When no single objective is honest — the Pareto frontier
# Twelve candidate price-service configurations score on margin and expected volume. Most are
# dominated (something else is better on both); what survives is the genuine trade-off menu to
# put in front of the decision-maker — multi-objective optimization without arbitrary weights.

# %%
configs = pl.DataFrame(
    {
        "config": [f"cfg_{i}" for i in range(12)],
        "margin": rng.uniform(8.0, 20.0, 12).round(1),
        "volume": rng.uniform(1000.0, 5000.0, 12).round(0),
    }
)
on_front = optimize.pareto_front(configs.select("margin", "volume").to_numpy(), maximize=True)
frontier = configs.with_columns(pl.Series("efficient", on_front))
print(frontier.sort("efficient", descending=True))
decision.pareto_frontier(
    frontier,
    x="margin",
    y="volume",
    efficient="efficient",
    label="config",
    title="The efficient frontier",
)

# %% [markdown]
# ## 6. Optimizing under uncertainty — mean vs worst case
# Capacity to commit before demand is known (build cost 6/unit, price 10). Optimizing the
# *expected* scenario over-commits relative to the robust answer; `criterion="worst"` is the
# max-min decision for when the bad future is unaffordable. The gap between the two answers is
# the price of robustness.

# %%
demand_scenarios = [{"demand": float(d)} for d in rng.normal(4000.0, 900.0, 60).clip(min=500)]


def capacity_value(x: np.ndarray, demand: float) -> float:
    built = float(x[0])
    return 10.0 * min(built, demand) - 6.0 * built


for criterion in ("mean", "worst"):
    solved = optimize.scenario_optimize(
        capacity_value,
        x0=[3000.0],
        scenarios=demand_scenarios,
        bounds=[(0.0, 8000.0)],
        criterion=criterion,
    )
    print(f"{criterion:5s}-case capacity: {float(solved.x[0]):7.0f} units")

# %% [markdown]
# ## 7. Network optimization — bottlenecks and backbones
# A two-warehouse distribution network: max-flow finds the throughput ceiling and *which* link
# is the binding bottleneck (the saturated cut); the minimum spanning tree is the cheapest
# backbone that still connects every site — the lower bound for network design.

# %%
lanes = pl.DataFrame(
    {
        "source": ["plant", "plant", "wh_north", "wh_north", "wh_south", "wh_south"],
        "target": ["wh_north", "wh_south", "store_a", "store_b", "store_b", "store_c"],
        "capacity": [400.0, 300.0, 250.0, 200.0, 250.0, 150.0],
    }
)
throughput, flows = graph.max_flow(lanes, origin="plant", sink="store_b", weight="capacity")
print(f"max units/day to store_b: {throughput:.0f}")
print(flows)

link_costs = lanes.rename({"capacity": "cost"}).with_columns(pl.col("cost") * 0.1)
backbone = graph.minimum_spanning_tree_edges(link_costs, weight="cost")
print(f"cheapest connected backbone: €{backbone['weight'].sum():.0f}/day in lane costs")

# %% [markdown]
# ## 8. Game theory — the competitor's optimizer runs too
# Two airlines choose hold-vs-discount. Each is individually better off discounting whatever the
# other does — so "both discount" is the unique Nash equilibrium even though "both hold" pays
# more: the pricing prisoner's dilemma. Knowing the equilibrium tells you which moves get
# matched (and what changing the *game* — capacity, loyalty, differentiation — is worth).

# %%
# payoffs in €m: (row = us, col = rival); strategies: 0 = hold price, 1 = discount
ours = np.array([[10.0, 4.0], [13.0, 6.0]])
theirs = np.array([[10.0, 13.0], [4.0, 6.0]])
print(f"pure Nash equilibria: {game.pure_nash(ours, theirs)}  (1,1) = both discount")
print(f"strategies surviving dominance: {game.iterated_dominance(ours, theirs)}")
print("equilibrium payoff 6 vs cooperative 10 -> the €4m gap is the prize for changing the game")


# %%
# Continuous price war: each side's best reply to the other's price (reaction functions from
# a linear-demand profit model). Best-response iteration settles where moves stop paying.
def our_reply(prices: np.ndarray) -> float:
    return 40.0 + 0.45 * prices[1]


def their_reply(prices: np.ndarray) -> float:
    return 35.0 + 0.50 * prices[0]


settle = game.best_response_dynamics([our_reply, their_reply], start=[120.0, 60.0])
print(
    f"price war settles at us=€{settle.point[0]:.0f}, rival=€{settle.point[1]:.0f} "
    f"after {settle.iterations} rounds (converged={settle.converged})"
)
war = pl.DataFrame(
    {
        "round": np.arange(settle.history.shape[0]),
        "us": settle.history[:, 0],
        "rival": settle.history[:, 1],
    }
)
interactive.line(
    war.unpivot(on=["us", "rival"], index="round"),
    "round",
    "value",
    color="variable",
    title="Best-response dynamics — where the price war stops",
)

# %% [markdown]
# **Takeaways:** the LP's shadow prices turn constraints into a shopping list — raising social's
# cap is worth 0.3 per euro while the budget itself only earns the marginal 1.1; the project
# knapsack beats greedy value-per-euro because all-or-nothing budgets aren't divisible; concave
# response moves the right answer from "all-in" to marginal-return-equalizing splits; the
# portfolio over-weights the diversifying bet relative to raw returns — covariance, not bet
# count, is what reduces risk; the Pareto frontier shrinks twelve configurations to the few
# genuinely competing ones; robust capacity sits far below the expected-value answer (that gap
# is the insurance premium); max-flow names the binding lane to upgrade; and the pricing game
# lands on mutual discounting at €6m against a cooperative €10m — which is precisely the
# argument for differentiation and loyalty: change the payoff matrix, not your move inside it.
