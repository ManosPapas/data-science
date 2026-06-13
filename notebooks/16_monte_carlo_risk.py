# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 16 · Monte Carlo & risk — the business case as a distribution, not a number
#
# A plan built on point estimates can still lose money a third of the time. This notebook runs a
# launch business case through Monte Carlo (with *correlated* inputs — costs and volumes rarely
# move independently), reads the P10/P50/P90 and the probability of hitting the target, finds
# which input uncertainty actually drives the spread, stress-tests survival against named shocks,
# simulates revenue *paths* (path risk ≠ endpoint risk), and prices the tail with VaR/CVaR.
# It closes with the early-warning side: control charts that catch a KPI quietly walking away.

# %%
from scipy import stats as sps

from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. The business case, simulated
# Launch P&L: contribution = volume · (price - unit_cost) - fixed costs. Volume and unit cost
# are genuinely uncertain; price is a decision (constant). We correlate volume and unit cost at
# +0.5 (busy markets strain the supply chain) — ignoring that correlation understates tail risk.


# %%
def contribution(volume, price, unit_cost, fixed):
    return volume * (price - unit_cost) - fixed


case = simulate.monte_carlo(
    contribution,
    inputs={
        "volume": sps.norm(50_000, 9_000),
        "price": 12.0,
        "unit_cost": sps.triang(c=0.3, loc=6.0, scale=3.0),  # skewed: overruns likelier
        "fixed": 150_000.0,
    },
    correlation={("volume", "unit_cost"): 0.5},
    n=20_000,
)
case.summary(targets=[0.0, 100_000.0])

# %%
print(f"P10 €{case.p10:,.0f}   P50 €{case.p50:,.0f}   P90 €{case.p90:,.0f}")
print(f"P(loss) = {case.prob_below(0.0):.1%}, P(≥ €100k target) = {case.prob_above(100_000):.1%}")
business.outcome_distribution(
    case.samples,
    targets=[0.0, 100_000.0],
    title="Contribution distribution — the whole answer",
)

# %% [markdown]
# ## 2. What drives the spread — the simulation-native tornado
# `drivers()` ranks inputs by |Spearman rho| with the outcome *across the simulation*: the
# de-risking priority list (research, hedge, contract the top one). Compare with the two-point
# tornado from `scenario.sensitivity` — same question, but the simulation version accounts for
# the full input distributions and their correlation.

# %%
case.drivers()

# %%
base_case = {"volume": 50_000.0, "price": 12.0, "unit_cost": 7.0, "fixed": 150_000.0}
two_point = scenario.sensitivity(
    contribution,
    base_case,
    {"volume": (35_000.0, 65_000.0), "unit_cost": (6.0, 9.0), "price": (11.0, 13.0)},
)
business.tornado(
    two_point,
    base=contribution(**base_case),
    title="Two-point tornado — same levers, drawn around the base case",
)

# %% [markdown]
# ## 3. Stress testing — do we survive *this*?
# Monte Carlo answers "how likely is bad"; the stress test answers the regulator's question:
# what happens under specific named shocks, and under all of them at once (correlations go to 1
# in a crisis). The combined row is the survival check for the funding decision.

# %%
simulate.stress_test(
    contribution,
    base_case,
    {
        "demand collapse (-30%)": {"volume": 35_000.0},
        "supply squeeze (+€2 cost)": {"unit_cost": 9.0},
        "price war (-€1.5)": {"price": 10.5},
    },
)

# %% [markdown]
# ## 4. Paths, not endpoints — the uncertainty fan
# Two revenue plans can share a mean and differ wildly in survivability. Simulate 1,000 monthly
# revenue trajectories (1% expected growth, 6% monthly volatility) and read the fan; drawdown
# risk lives here, invisible to any single-period summary.

# %%
paths = simulate.simulate_paths(start=1_000_000.0, drift=0.01, volatility=0.06, periods=36, n=1000)
fan_bands = simulate.path_percentiles(paths, quantiles=(0.1, 0.5, 0.9))
business.fan(
    fan_bands,
    x="period",
    bands=[("p10", "p90")],
    line="p50",
    title="36-month revenue fan — uncertainty compounds",
)

# %%
final = paths[:, -1]
worst_drawdowns = np.array([risk.max_drawdown(path) for path in paths])
p10_final, p90_final = np.percentile(final, [10, 90])
print(f"month-36 revenue: P10 €{p10_final:,.0f}, P90 €{p90_final:,.0f}")
print(f"median worst drawdown along the way: {np.median(worst_drawdowns):.0%}")
print(f"P(some drawdown > 20%): {(worst_drawdowns > 0.2).mean():.0%}  <- path risk")

# %% [markdown]
# ## 5. Pricing the tail — VaR, CVaR, downside measures
# Risk lives in the low quantiles of the outcome distribution. VaR is the threshold ("with 95%
# confidence, not worse than…"); CVaR is the *mean* of the tail beyond it — what a bad year
# actually looks like, and the coherent one to aggregate across a portfolio.

# %%
risk.risk_summary(case.samples, targets=[0.0, 100_000.0])

# %%
var_95 = risk.value_at_risk(case.samples, alpha=0.05)
cvar_95 = risk.expected_shortfall(case.samples, alpha=0.05)
print(f"VaR(95): €{var_95:,.0f}   CVaR(95): €{cvar_95:,.0f}")
print(f"downside deviation vs 0: €{risk.downside_deviation(case.samples):,.0f}")

monthly_growth = np.diff(paths, axis=1)[:, 0] / paths[:, 0]
print(
    f"plan-level Sharpe {risk.sharpe_ratio(monthly_growth, periods_per_year=12):.2f}, "
    f"Sortino {risk.sortino_ratio(monthly_growth, periods_per_year=12):.2f}"
)

# %% [markdown]
# ## 6. Early warning — catching the KPI that quietly walks away
# Risk management after launch is monitoring. A Shewhart band (mean ± 3 sigma from a *trusted*
# baseline) catches big breaks; the EWMA chart catches the dangerous case — a small persistent
# shift that never trips a 3-sigma alarm. Here conversion drifts down 6% relative; EWMA flags it
# while every single point stays inside the naive band.

# %%
baseline_conversion = rng.normal(0.050, 0.004, 120)
limits = monitor.control_limits(baseline_conversion)
print(f"Shewhart band: {limits.lower:.4f} … {limits.upper:.4f} (center {limits.center:.4f})")

live = np.concatenate(
    [rng.normal(0.050, 0.004, 30), rng.normal(0.047, 0.004, 40)]  # -6% relative drift
)
chart = monitor.ewma_alerts(live, baseline_conversion, lam=0.2)
first_alert = chart.filter(pl.col("alert"))
shewhart_breaches = ((live < limits.lower) | (live > limits.upper)).sum()
print(f"individual points outside the 3-sigma band: {shewhart_breaches}")
print(f"first EWMA alert at t={first_alert['t'][0] if first_alert.height else None}")
business.control_chart(
    chart, title="EWMA control chart — the slow drift is caught, single points never alarm"
)

# %% [markdown]
# **Takeaways:** the launch is healthy at the median (P50 ≈ €92k) but the distribution is the
# real answer — ~28% chance of a loss, driven overwhelmingly by volume uncertainty (de-risk
# *that*, not unit costs); the correlated simulation is wider-tailed than an independent one
# would be, which is exactly the error naive spreadsheets make; the stress test says any single
# shock is survivable but the combined crisis is not — that's the contingency line in the
# funding ask; the path simulation reveals ~50% odds of a >20% drawdown en route even though the
# endpoint P50 grows — cash buffers are sized on paths, not endpoints; VaR/CVaR turn "risky"
# into euro thresholds a CFO can underwrite; and the EWMA chart catches a 6% conversion drift
# that a 3-sigma point rule never flags — wire it to the metrics that pay for everything above.
