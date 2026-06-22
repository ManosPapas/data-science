# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Pax-revenue · Phase 6 — Oil × FX × airspace: margin at risk
#
# **The decision this whole project builds to:** with fuel the dominant cost on these five
# long-hauls (Phase 3), revenue in five currencies (Phase 2), demand we can bound (Phase 4) and a
# fare lever we can't yet trust to a number (Phase 5) — **how exposed is contribution to an oil
# spike, a currency move, or a Middle-East airspace closure, and how big a fare move would it take to
# hold the line?**
#
# We model **annual fuel-contribution** = fare revenue − fuel cost (the oil-exposed slice of P&L),
# and push it through a **correlated Monte-Carlo** (Brent→jet fuel, FX on NOK/SEK/ZAR/GBP/EUR,
# demand), then deterministic **stress tests** and **scenarios**, and close with the per-route fare
# uplift needed to pass through an oil spike.
#
# > Input: `data/processed/pax_revenue_enriched.parquet`. All macro inputs are documented assumptions.

# %%
from scipy import stats as sps

from core.config import ROOT
from core.prelude import *

set_theme()
ENRICHED = ROOT / "data" / "processed" / "pax_revenue_enriched.parquet"
fares = read_parquet(ENRICHED).filter(~pl.col("is_refund"))

# %% [markdown]
# ## 1. Baseline economics (from the data) and the value function
# Per-currency annual revenue (already in baseline-USD), and annual fuel tonnes split by whether the
# route is Middle-East-airspace-exposed. Jet fuel is mapped off Brent (`jet $/t ≈ 9·Brent + 80`,
# calibrated so Brent $80 → $800/t). Demand scales **revenue only** — you burn the fuel whether the
# cabin is full or not, which is exactly why a demand slump bites contribution so hard.

# %%
REV = {c: 0.0 for c in ["USD", "GBP", "EUR", "NOK", "SEK", "ZAR"]}
for ccy, amt in fares.group_by("currency").agg(pl.col("charge_usd").sum()).iter_rows():
    REV[ccy] = float(amt)

rt = (
    fares.group_by("route").agg(
        pl.col("leg_id").n_unique().alias("legs"),
        pl.col("fuel_tonnes").first().alias("tonnes_per_leg"),
        pl.col("me_airspace_exposed").first().alias("exposed"),
    )
    .with_columns((pl.col("legs") * pl.col("tonnes_per_leg")).alias("annual_tonnes"))
)
TONNES_EXP = float(rt.filter(pl.col("exposed"))["annual_tonnes"].sum())
TONNES_UNEXP = float(rt.filter(~pl.col("exposed"))["annual_tonnes"].sum())
print(f"annual revenue ${sum(REV.values()):,.0f} | fuel tonnes {TONNES_EXP + TONNES_UNEXP:,.0f} "
      f"(exposed {TONNES_EXP:,.0f})")


def net_contribution(*, brent, nok_mult, sek_mult, zar_mult, gbp_mult, eur_mult,
                     demand_mult, reroute_factor):
    """Annual fuel-contribution (USD): fare revenue (FX- & demand-adjusted) − fuel cost."""
    jet = 9.0 * brent + 80.0
    revenue = demand_mult * (
        REV["USD"] + nok_mult * REV["NOK"] + sek_mult * REV["SEK"] + zar_mult * REV["ZAR"]
        + gbp_mult * REV["GBP"] + eur_mult * REV["EUR"]
    )
    fuel = jet * (TONNES_UNEXP + reroute_factor * TONNES_EXP)
    return revenue - fuel


BASE = {"brent": 80.0, "nok_mult": 1.0, "sek_mult": 1.0, "zar_mult": 1.0, "gbp_mult": 1.0,
        "eur_mult": 1.0, "demand_mult": 1.0, "reroute_factor": 1.0}
baseline_net = float(net_contribution(**BASE))
print(f"baseline annual fuel-contribution: ${baseline_net:,.0f}")

# %% [markdown]
# ## 2. Monte-Carlo: the contribution distribution under correlated macro risk
# Brent, FX and demand sampled together — **NOK rises with oil** (petro-currency, +0.5), **ZAR falls**
# (risk-off, −0.35), and **high oil drags demand** (−0.2). Independence would understate the tail;
# the copula couples them.

# %%
inputs = {
    "brent": sps.norm(80, 12),
    "nok_mult": sps.norm(1.0, 0.08),
    "sek_mult": sps.norm(1.0, 0.08),
    "zar_mult": sps.norm(1.0, 0.12),
    "gbp_mult": sps.norm(1.0, 0.05),
    "eur_mult": sps.norm(1.0, 0.05),
    "demand_mult": sps.norm(1.0, 0.08),
    "reroute_factor": 1.0,  # constant here; stressed in §4
}
correlation = {
    ("brent", "nok_mult"): 0.5, ("brent", "zar_mult"): -0.35,
    ("brent", "eur_mult"): 0.1, ("brent", "demand_mult"): -0.2,
}
sim = simulate.monte_carlo(net_contribution, inputs, n=20000, correlation=correlation)

print(sim.summary(targets=baseline_net))
print("\nrisk:", risk.risk_summary(sim.samples, targets=baseline_net))
print(f"\nVaR(5%)  = ${risk.value_at_risk(sim.samples):,.0f}  (1-in-20 bad-year contribution floor)")
print(f"CVaR(5%) = ${risk.expected_shortfall(sim.samples):,.0f}  (mean of the worst 5%)")
print(f"P(below baseline) = {sim.prob_below(baseline_net):.0%}")

# %%
fig, axes = base.grid(1, ncols=1, figsize=(11, 5))
business.outcome_distribution(sim.samples, percentiles=(10, 50, 90), targets=[baseline_net],
                              ax=axes[0], title="Annual fuel-contribution — Monte-Carlo (95% range)")

# %% [markdown]
# ## 3. What drives the spread? (simulation-native tornado)
# `|Spearman|` of each uncertain input with the outcome — de-risk (hedge fuel, contract FX) the top
# driver first. Note the FX inputs' signs reflect their **correlation with oil** (NOK reads negative
# because it co-moves with Brent) — association across draws, not a direct effect, so hedge oil and
# FX *together*.

# %%
drivers = sim.drivers()
print(drivers)
fig, axes = base.grid(1, ncols=1, figsize=(9, 4.5))
axes[0].barh(drivers["input"], drivers["spearman"],
             color=["tab:red" if v < 0 else "tab:green" for v in drivers["spearman"]])
axes[0].axvline(0, color="#444", lw=1)
axes[0].set(title="Contribution drivers (Spearman with outcome)", xlabel="rho")
axes[0].invert_yaxis()

# %% [markdown]
# ## 4. Deterministic stress tests & coherent scenarios
# The simulation says how *likely* bad outcomes are; stress tests say whether we *survive* specific
# ones. The combined row is the all-at-once crisis (correlations → 1).

# %%
stresses = {
    "oil spike $110": {"brent": 110.0},
    "FX weakness -12%": {"nok_mult": 0.88, "sek_mult": 0.88, "zar_mult": 0.85,
                          "eur_mult": 0.94, "gbp_mult": 0.94},
    "demand slump -15%": {"demand_mult": 0.85},
    "ME airspace closure": {"reroute_factor": 1.12},
}
stress = simulate.stress_test(net_contribution, BASE, stresses).with_columns(
    (pl.col("value") / baseline_net - 1).round(3).alias("vs_base_pct")
)
stress

# %%
scenarios = {
    "downturn": {"brent": 100.0, "demand_mult": 0.90, "nok_mult": 1.05, "zar_mult": 0.90},
    "optimistic": {"brent": 65.0, "demand_mult": 1.05, "nok_mult": 0.97},
    "stagflation": {"brent": 115.0, "demand_mult": 0.85, "zar_mult": 0.85, "reroute_factor": 1.10},
}
scenario.scenario_table(net_contribution, BASE, scenarios)

# %%
# One-at-a-time sensitivity -> the classic tornado (biggest lever on top).
ranges = {
    "brent": (60.0, 110.0), "demand_mult": (0.90, 1.10), "nok_mult": (0.90, 1.10),
    "zar_mult": (0.85, 1.15), "reroute_factor": (1.0, 1.12),
}
sens = scenario.sensitivity(net_contribution, BASE, ranges)
fig, axes = base.grid(1, ncols=1, figsize=(9, 4.5))
business.tornado(sens, base=baseline_net, ax=axes[0], title="Contribution sensitivity (low→high input)")

# %% [markdown]
# ## 5. The closing question: fare uplift to pass through an oil spike
# If Brent jumps to $110 (jet $1,070/t), each route's fuel bill rises by `Δjet × tonnes`. Holding
# load, the **fare uplift per seat** that fully recovers it is `extra fuel per seat ÷ current fare`.
# Whether the market *accepts* that uplift is the elasticity question Phase 5 showed this data can't
# answer — so these are the **pass-through targets a price test must validate**, not guaranteed wins.

# %%
DELTA_JET = (9.0 * 110.0 + 80.0) - 800.0  # $/tonne increase, baseline $800 -> $1,070
route_tbl = (
    fares.group_by("route").agg(
        pl.col("charge_usd").mean().alias("avg_fare"),
        pl.col("fuel_tonnes").first().alias("tonnes_per_leg"),
        pl.col("me_airspace_exposed").first().alias("exposed"),
    )
    .join(
        fares.group_by("route", "leg_id").agg(pl.len().alias("pax"))
        .group_by("route").agg(pl.col("pax").mean().alias("avg_pax")),
        on="route",
    )
    .with_columns((pl.col("tonnes_per_leg") * DELTA_JET).alias("extra_fuel_per_leg"))
    .with_columns((pl.col("extra_fuel_per_leg") / pl.col("avg_pax")).alias("extra_per_seat"))
    .with_columns((pl.col("extra_per_seat") / pl.col("avg_fare") * 100).round(1).alias("uplift_pct"))
    .sort("uplift_pct", descending=True)
)
route_tbl.select("route", "avg_fare", "avg_pax", "extra_per_seat", "uplift_pct", "exposed")

# %%
fig, axes = base.grid(1, ncols=1, figsize=(11, 5))
colors = ["tab:red" if e else "tab:blue" for e in route_tbl["exposed"]]
axes[0].bar(route_tbl["route"], route_tbl["uplift_pct"], color=colors, alpha=0.85)
axes[0].set(title="Fare uplift to pass through Brent $110 (red = ME-airspace-exposed)",
            ylabel="% fare increase")
axes[0].tick_params(axis="x", rotation=45)

# %% [markdown]
# **Takeaways — the whole project, landed**
#
# - **Margin at risk is real but survivable in the central case.** Baseline annual fuel-contribution
#   is the §1 figure; the Monte-Carlo P10 / VaR(5%) / CVaR(5%) in §2 quantify the bad-year floor, and
#   `P(below baseline)` is the odds of missing plan from macro alone.
# - **Demand is the top driver (Spearman +0.77), Brent a close second (−0.70)** (§3): because fuel is
#   fixed per flight, every point of load flows straight to contribution. So the first levers are
#   **demand protection / forecasting** and **fuel hedging**; FX signs are entangled with oil via the
#   copula, so hedge them together.
# - **Survivable singly, not jointly** (§4): an oil spike to $110 cuts contribution ~48%, a 15%
#   demand slump ~37%, FX weakness ~16%, and the ME-airspace closure only ~9% (it hits just the
#   exposed routes' fuel). But the **combined crisis turns the $16M contribution negative (≈ −$1.6M)**
#   — the number to capitalise / hedge against; the geopolitics shock is the smallest single piece.
# - **Pass-through targets (§5):** absorbing Brent $110 needs an **~8% (ARN-HKT) to ~28% (CPT-LGW)**
#   fare uplift, heaviest on the **low-yield, low-load inbound legs** (CPT-LGW 27%, BKK-OSL 26%) — the
#   same legs Phase 3 flagged as thin-cushion. **But** whether that uplift sticks is the elasticity
#   question Phase 5 proved this data can't settle — so the action is a **fuel-exposure-prioritised
#   price test**, with fuel hedging in the meantime.
# - **End to end:** raw feed → clean FX-normalised table → fuel economics → demand forecast → price
#   response → this risk model. Each phase is one runnable notebook on the five fuel-/airspace-exposed
#   long-hauls; every external assumption is documented and feed-replaceable.
