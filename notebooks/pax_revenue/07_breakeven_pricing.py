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
# # Pax-revenue · Phase 7 — Break-even fare per product class
#
# **The ask:** a mathematical rule for what each product class (EL/EC/EP/PL/PC/PP) must charge so a
# flight makes *at least* a tiny profit — respecting (a) **how we sell** (a fare ladder the RM system
# fills along, so the constraint is on each class's *average* realized fare) and (b) **how classes
# are distributed today** (the mix and the price ratios between brands).
#
# ## The model
# For one flight on a route:
# - capacity `C`, expected load `L` → sold seats `S = C·L`
# - product-class mix shares `w_k` (Σ w_k = 1) → seats sold in class k: `s_k = S·w_k`
# - the **brand price ladder** `r_k` = class k's fare ÷ EL's fare (so EL = 1, premium > 1) — keeps
#   the relative structure intact (you can't flat-price brands that mean different things)
# - flight cost `K` and a tiny target margin `ε`
#
# Hold the ladder and scale it by one factor `α` until revenue covers cost:
# > revenue `= Σ_k s_k·p_k = Σ_k s_k·(α·r_k) = α·S·Σ_k w_k r_k`  set `= K·(1+ε)`
# >
# > **`α* = K(1+ε) / (S · Σ_k w_k r_k)`**   and   **`p_k* = α*·r_k`**
#
# The mix-weighted break-even fare is just `p̄* = K(1+ε)/S` (cost per sold seat + margin); each class
# is that, re-spread by its brand ratio. `p_k*` is a **floor on the class average**, not a single
# price — the RM ladder still sells cheaper early and dearer late around it.
#
# > Input: `data/processed/pax_revenue_enriched.parquet`. Cost beyond fuel and the +25% ancillary
# > revenue uplift are documented assumptions.

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
ENRICHED = ROOT / "data" / "processed" / "pax_revenue_enriched.parquet"

CLASSES = ["EL", "EC", "EP", "PL", "PC", "PP"]
CAPACITY = 338
FUEL_SHARE = 0.32       # fuel as a share of total operating cost (long-haul ~30-40%); documented
MARGIN = 0.01           # the "tiny profit" target: 1% over cost
ANCILLARY_UPLIFT = 0.25  # ancillary revenue (bags/seats/bundles/cargo) as a fraction of fare revenue

fares = (
    read_parquet(ENRICHED)
    .filter(~pl.col("is_refund"))
    .filter(pl.col("fare_brand").is_in(CLASSES))
)
print(f"{fares.height:,} priced fares across {fares['route'].n_unique()} routes, {len(CLASSES)} classes")

# %% [markdown]
# ## 1. The brand price ladder (ratios) — how classes are priced relative to each other
# Pooled across the subset (stable even where a class is thin on one route); EL = 1.0.

# %%
class_avg = (
    fares.group_by("fare_brand").agg(pl.col("charge_usd").mean().alias("avg_usd"), pl.len().alias("n"))
)
el_avg = float(class_avg.filter(pl.col("fare_brand") == "EL")["avg_usd"][0])
RATIOS = {row["fare_brand"]: row["avg_usd"] / el_avg for row in class_avg.iter_rows(named=True)}
RATIOS = {k: round(RATIOS[k], 3) for k in CLASSES}
print("brand ratios (vs EL):", RATIOS)

# %% [markdown]
# ## 2. Per-route inputs: mix, sold seats, and cost
# Mix `w_k` and sold seats `S` come straight from the data (current behaviour). Fuel cost per leg is
# the Phase-3 figure; full operating cost = fuel ÷ `FUEL_SHARE`.

# %%
mix = (
    fares.group_by("route", "fare_brand").agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum().over("route")).alias("w"))
)
mix_wide = mix.pivot(values="w", index="route", on="fare_brand").fill_null(0.0)

route_inputs = (
    fares.group_by("route").agg(
        pl.col("fuel_cost_usd").first().alias("fuel_per_leg"),
        pl.col("charge_usd").mean().round(0).alias("cur_avg_fare"),
        pl.col("me_airspace_exposed").first().alias("exposed"),
    )
    .join(
        fares.group_by("route", "leg_id").agg(pl.len().alias("pax"))
        .group_by("route").agg(pl.col("pax").mean().round(0).alias("sold_seats")),
        on="route",
    )
    .with_columns((pl.col("fuel_per_leg") / FUEL_SHARE).round(0).alias("full_cost"))
    .sort("route")
)
route_inputs

# %% [markdown]
# ## 3. Break-even fares per route
# Apply the rule, **crediting +25% ancillary revenue** (fares then only need to cover cost ÷ 1.25).
# We show the floor at **full operating cost** (the real target) and at **fuel-only** (rock bottom —
# below this the flight can't even cover the gas).

# %%
def breakeven_fares(cost, sold_seats, w, ratios, margin, revenue_uplift=0.0):
    """Minimum class-average *fares* covering cost(1+margin), crediting ancillary ``revenue_uplift``.

    Ancillaries add ``revenue_uplift`` × fare revenue, so the fare only needs to cover
    ``cost(1+margin)/(1+revenue_uplift)``.
    """
    denom = sum(w.get(k, 0.0) * ratios[k] for k in ratios)
    alpha = cost * (1.0 + margin) / ((1.0 + revenue_uplift) * sold_seats * denom)
    return {k: alpha * ratios[k] for k in ratios}


rows = []
for r in route_inputs.iter_rows(named=True):
    w = {k: float(mix_wide.filter(pl.col("route") == r["route"])[k][0]) for k in CLASSES}
    be_full = breakeven_fares(r["full_cost"], r["sold_seats"], w, RATIOS, MARGIN, ANCILLARY_UPLIFT)
    be_fuel = breakeven_fares(r["fuel_per_leg"], r["sold_seats"], w, RATIOS, 0.0, ANCILLARY_UPLIFT)
    pbar_fare = r["full_cost"] * (1 + MARGIN) / ((1 + ANCILLARY_UPLIFT) * r["sold_seats"])
    total_rev = r["sold_seats"] * r["cur_avg_fare"] * (1 + ANCILLARY_UPLIFT)
    rows.append({
        "route": r["route"], "sold_seats": r["sold_seats"], "cur_avg_fare": r["cur_avg_fare"],
        "breakeven_avg_fare": round(pbar_fare, 0),
        "surplus_per_flight": round(total_rev - r["full_cost"], 0),
        "ladder_uplift_needed_pct": round((pbar_fare / r["cur_avg_fare"] - 1) * 100, 1),
        **{f"be_{k}": round(be_full[k], 0) for k in CLASSES},
        "fuelfloor_EL": round(be_fuel["EL"], 0),
    })
breakeven = pl.DataFrame(rows).sort("ladder_uplift_needed_pct", descending=True)
breakeven.select("route", "sold_seats", "cur_avg_fare", "breakeven_avg_fare",
                 "surplus_per_flight", "ladder_uplift_needed_pct")

# %% [markdown]
# ## 4. The recommended minimum fare per product class (full-cost break-even)
# This is the deliverable: the floor each class average must clear, per route.

# %%
breakeven.select("route", *[f"be_{k}" for k in CLASSES])

# %% [markdown]
# ## 5. Current vs break-even, per class (busiest route) + load sensitivity
# Left: where today's class averages sit against the break-even floor. Right: the floor falls as the
# flight fills — the lever isn't only price, it's **load** (empty seats raise everyone's break-even).

# %%
R = breakeven.sort("sold_seats", descending=True)["route"][0]
ri = route_inputs.filter(pl.col("route") == R).to_dicts()[0]
w_r = {k: float(mix_wide.filter(pl.col("route") == R)[k][0]) for k in CLASSES}
cur_by_class = {row["fare_brand"]: row["avg_usd"] for row in
                fares.filter(pl.col("route") == R).group_by("fare_brand")
                .agg(pl.col("charge_usd").mean().alias("avg_usd")).iter_rows(named=True)}
be_r = breakeven_fares(ri["full_cost"], ri["sold_seats"], w_r, RATIOS, MARGIN, ANCILLARY_UPLIFT)

fig, axes = base.grid(2, figsize=(13, 5))
x = np.arange(len(CLASSES))
axes[0].bar(x - 0.2, [cur_by_class.get(k, 0) for k in CLASSES], 0.4, label="current avg", color="tab:blue")
axes[0].bar(x + 0.2, [be_r[k] for k in CLASSES], 0.4, label="break-even floor", color="tab:red", alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(CLASSES)
axes[0].set(title=f"{R}: current vs break-even fare by class", ylabel="USD"); axes[0].legend()

loads = np.arange(0.45, 1.01, 0.05)
for k in ["EL", "EC", "PC"]:
    be_at_load = [breakeven_fares(ri["full_cost"], CAPACITY * L, w_r, RATIOS, MARGIN, ANCILLARY_UPLIFT)[k] for L in loads]
    axes[1].plot(loads * 100, be_at_load, marker="o", label=k)
axes[1].axvline(ri["sold_seats"] / CAPACITY * 100, ls="--", color="#888", label="current load")
axes[1].set(title=f"{R}: break-even fare vs load factor", xlabel="load factor (%)", ylabel="USD")
axes[1].legend()

# %% [markdown]
# ## 6. The honest unit: round-trip (O&D pair), and how it all hinges on the cost assumption
# Per-*leg* fares are distorted by how a round-trip fare is **split across the outbound and inbound
# segments** — the inbound leg looks poorer than it is. Netting both directions into the O&D pair
# removes that artifact. And the whole verdict scales with `FUEL_SHARE`, so we sweep it.

# %%
pair = (
    fares.group_by("route", "od_pair").agg(
        pl.col("charge_usd").sum().alias("fare_rev"),
        pl.col("leg_id").n_unique().alias("legs"),
        pl.col("fuel_cost_usd").first().alias("fuel_leg"),
    )
    .with_columns((pl.col("legs") * pl.col("fuel_leg") / FUEL_SHARE).alias("full_cost"))
    .group_by("od_pair").agg(pl.col("fare_rev").sum(), pl.col("full_cost").sum(), pl.col("legs").sum())
    .with_columns((pl.col("fare_rev") * (1 + ANCILLARY_UPLIFT)).round(0).alias("total_rev"))
    .with_columns(
        (pl.col("total_rev") / pl.col("full_cost")).round(2).alias("rev_cost_ratio"),
        ((pl.col("full_cost") / pl.col("total_rev") - 1) * 100).round(0).alias("uplift_needed_pct"),
    )
    .sort("rev_cost_ratio")
)
pair

# %%
fuel_total = (
    fares.group_by("route").agg(pl.col("leg_id").n_unique().alias("L"),
                                pl.col("fuel_cost_usd").first().alias("fl"))
    .with_columns((pl.col("L") * pl.col("fl")).alias("fuel"))
)["fuel"].sum()
total_rev = fares["charge_usd"].sum() * (1 + ANCILLARY_UPLIFT)
sweep = pl.DataFrame({"fuel_share": [0.25, 0.32, 0.40, 0.50]}).with_columns(
    (total_rev / (fuel_total / pl.col("fuel_share"))).round(2).alias("portfolio_rev_cost_ratio")
)
print(f"fare + {ANCILLARY_UPLIFT:.0%} ancillary revenue vs modeled full opex (whole subset):")
sweep

# %% [markdown]
# **Takeaways — the rule, and the truth with ancillaries in**
#
# - **The deliverable (the rule):** `p_k* = [K(1+ε)/((1+a)·S)] · r_k / Σ_j w_j r_j` — cost per sold
#   seat (+1% margin, less the ancillary credit `a`=25%), re-spread by the brand ladder. §4 is the
#   per-route, per-class fare floor; drop it into the RM system as a hard minimum on each class's
#   *average* realized fare. Ancillaries lower every floor ~20% vs fare-only.
# - **Even with +25% ancillaries, most legs are still short on full cost.** 3 of 10 directional legs
#   clear it — the high-yield outbound ARN-BKK / OSL-HKT / ARN-HKT (already 23–36% above their floor).
#   The other 7 need fare uplifts of ~38% (LGW-CPT) up to ~106% (CPT-LGW).
# - **Round-trip is the honest unit, and it splits the network in two:** the Phuket pairs turn
#   profitable (HKT-OSL rev/cost **1.04**, ARN-HKT **1.13**), ARN-BKK breaks even (**0.97**) — but the
#   two big-volume pairs **CPT-LGW (0.61)** and **BKK-OSL (0.60)** still lose ~40%, dragging the
#   **portfolio to 0.68**. Those two pairs are the problem, and it's **low load × low yield**, not headline fare.
# - **The honest hinge:** the verdict rides on `FUEL_SHARE` — with ancillaries in, the portfolio only
#   breaks even if fuel is ~half of total opex (sweep: 0.53 → 1.05 across fuel-share 0.25 → 0.50). So
#   **load real opex** before treating any absolute level as gospel; the *formula* is the durable part.
# - **What to actually do:** (1) use §4 as a **fare floor / RM guardrail** (never sell a class below
#   it), hardest on the CPT-LGW and BKK-OSL pairs; (2) attack those two pairs on **load** (fill the
#   empty inbound legs) and **ancillary attach**, not just fare; (3) size any fare *uplift* with the
#   **price test** from Phase 5 — raising fares shifts the mix `w_k` this rule holds fixed.
