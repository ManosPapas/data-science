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
# # Pax-revenue · Phase 3 — Enrich: distance → fuel cost → oil exposure
#
# **Decision this informs:** which of the five long-hauls is most exposed when jet fuel rises or
# Middle-East airspace closes — and what fare each route must earn to cover its fuel. This is the
# bridge between the revenue data (Phases 1-2) and the oil/geopolitics story (Phase 6).
#
# The raw feed has **no cost column**, so fuel economics come from a small, fully **documented**
# engineering model: great-circle distance (`features.geo.haversine`) → block hours → 787-9 fuel
# burn → fuel cost per flight and per seat → the **fuel break-even load factor**. Every constant
# below is a labelled assumption, structured to be replaced by ops/finance actuals.
#
# > Input: `data/processed/pax_revenue_clean.parquet` · Output: `..._enriched.parquet`

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
CLEAN = ROOT / "data" / "processed" / "pax_revenue_clean.parquet"
ENRICHED = ROOT / "data" / "processed" / "pax_revenue_enriched.parquet"

clean = read_parquet(CLEAN)
fares = clean.filter(~pl.col("is_refund"))
print(f"loaded {clean.height:,} rows ({fares.height:,} paid fares)")

# %% [markdown]
# ## 1. Assumptions (documented — swap for actuals before any hard financial call)
# A Boeing 787-9 (the fleet in this data, capacity 338). Burn and price are round, defensible
# figures; the reroute penalty captures the extra block time Europe↔Asia sectors fly when they
# detour around closed Middle-East airspace.

# %%
AIRPORTS = {  # (lat, lon) in degrees
    "OSL": (60.1976, 11.1004), "ARN": (59.6519, 17.9186), "LGW": (51.1537, -0.1821),
    "BKK": (13.6900, 100.7501), "HKT": (8.1132, 98.3170), "CPT": (-33.9715, 18.6021),
}
CAPACITY = 338            # seats (787-9; dominant FlCapacity in the feed)
CRUISE_KMH = 900.0        # ~Mach 0.85
OVERHEAD_HR = 0.75        # taxi + climb + descent, per flight
BURN_KG_PER_HR = 5600.0   # 787-9 cruise fuel burn
JET_USD_PER_TONNE = 800.0 # baseline jet-fuel price (Phase 6 varies this with Brent)
REROUTE_HR = 0.8          # extra block time on ME-airspace-exposed sectors (avoidance detour)

# %% [markdown]
# ## 2. Build the route reference table
# One row per directional route: distance, block hours (with the reroute add-on where it applies),
# fuel burnt, fuel cost, and fuel cost per seat.

# %%
route_ref = clean.select(
    "route_code", "route", "od_pair", "origin", "dest", "direction", "market",
    "me_airspace_exposed",
).unique()

lat = {k: v[0] for k, v in AIRPORTS.items()}
lon = {k: v[1] for k, v in AIRPORTS.items()}
route_ref = route_ref.with_columns(
    pl.col("origin").replace_strict(lat, return_dtype=pl.Float64).alias("o_lat"),
    pl.col("origin").replace_strict(lon, return_dtype=pl.Float64).alias("o_lon"),
    pl.col("dest").replace_strict(lat, return_dtype=pl.Float64).alias("d_lat"),
    pl.col("dest").replace_strict(lon, return_dtype=pl.Float64).alias("d_lon"),
)
route_ref = route_ref.with_columns(
    pl.Series(
        "distance_km",
        geo.haversine(
            route_ref["o_lat"].to_numpy(), route_ref["o_lon"].to_numpy(),
            route_ref["d_lat"].to_numpy(), route_ref["d_lon"].to_numpy(),
        ),
    ).round(0)
)

route_ref = route_ref.with_columns(
    (pl.col("distance_km") / CRUISE_KMH + OVERHEAD_HR
     + pl.when("me_airspace_exposed").then(REROUTE_HR).otherwise(0.0)).round(2).alias("block_hours"),
).with_columns(
    (pl.col("block_hours") * BURN_KG_PER_HR / 1000).round(1).alias("fuel_tonnes"),
).with_columns(
    (pl.col("fuel_tonnes") * JET_USD_PER_TONNE).round(0).alias("fuel_cost_usd"),
    (pl.col("fuel_tonnes") * JET_USD_PER_TONNE / CAPACITY).round(1).alias("fuel_cost_per_seat"),
)
route_ref.select("route", "distance_km", "block_hours", "fuel_tonnes", "fuel_cost_usd",
                 "fuel_cost_per_seat", "me_airspace_exposed").sort("distance_km", descending=True)

# %% [markdown]
# ## 3. Join observed economics → the fuel break-even table
# Bring in each route's **observed average fare** (USD) and **average final load factor** (per leg),
# then derive the numbers that matter:
# - `fuel_breakeven_lf` — the load factor at which fare revenue just covers **fuel** (a lower bound
#   on total break-even; we only model the oil-exposed cost here).
# - `cushion_pts` — observed final LF minus that break-even: the fuel-margin buffer.
# - `fuel_share_of_rev` — fuel cost as a share of fare revenue at the observed fill.

# %%
# Final load factor per leg, then averaged per route (leg-level, so it isn't pax-weighted).
leg_lf = (
    fares.group_by("route_code", "leg_id")
    .agg(pl.col("leg_final_lf").first().alias("flf"))
    .group_by("route_code").agg(pl.col("flf").mean().round(1).alias("avg_final_lf"))
)
obs = (
    fares.group_by("route_code")
    .agg(
        pl.col("charge_usd").mean().round(0).alias("avg_fare_usd"),
        pl.col("charge_usd").sum().round(0).alias("revenue_usd"),
        pl.col("leg_id").n_unique().alias("legs"),
    )
    .join(leg_lf, on="route_code")
)

econ = (
    route_ref.join(obs, on="route_code")
    .with_columns(
        (pl.col("fuel_cost_usd") / (pl.col("avg_fare_usd") * CAPACITY) * 100).round(1).alias("fuel_breakeven_lf"),
        (pl.col("avg_fare_usd") * CAPACITY * pl.col("avg_final_lf") / 100).round(0).alias("revenue_per_flight"),
    )
    .with_columns(
        (pl.col("avg_final_lf") - pl.col("fuel_breakeven_lf")).round(1).alias("cushion_pts"),
        (pl.col("fuel_cost_usd") / pl.col("revenue_per_flight") * 100).round(1).alias("fuel_share_of_rev"),
    )
)
econ.select("route", "distance_km", "avg_fare_usd", "avg_final_lf", "fuel_cost_per_seat",
            "fuel_breakeven_lf", "cushion_pts", "fuel_share_of_rev").sort(
    "fuel_share_of_rev", descending=True)

# %% [markdown]
# ## 4. Does fare track distance? (it should, if pricing reflects cost)
# A weak distance→fare link is itself a finding: fares set by demand/competition, not cost — which
# is exactly why an oil shock squeezes margin rather than passing through to price automatically.

# %%
fig, axes = base.grid(2)
eda.scatter(econ, "distance_km", "avg_fare_usd", hue="market", ax=axes[0],
            title="Avg fare vs great-circle distance")
eda.scatter(econ, "distance_km", "fuel_cost_per_seat", ax=axes[1],
            title="Fuel cost per seat vs distance")
print(stats.correlation_test(econ["distance_km"].to_numpy(), econ["avg_fare_usd"].to_numpy(),
                             method="spearman"))

# %% [markdown]
# ## 5. The oil-exposure ranking
# Fuel as a share of fare revenue, and the fuel-margin cushion, per route — who hurts first when
# Brent climbs.

# %%
ranked = econ.sort("fuel_share_of_rev", descending=True)
fig, axes = base.grid(2, figsize=(13, 5))
axes[0].barh(ranked["route"], ranked["fuel_share_of_rev"], color="tab:red")
axes[0].invert_yaxis(); axes[0].set_title("Fuel as % of fare revenue"); axes[0].set_xlabel("%")
cushion = econ.sort("cushion_pts")
colors = ["tab:red" if c < 25 else "tab:green" for c in cushion["cushion_pts"]]
axes[1].barh(cushion["route"], cushion["cushion_pts"], color=colors)
axes[1].invert_yaxis(); axes[1].set_title("Fuel-margin cushion (obs LF − break-even LF, pts)")
axes[1].set_xlabel("load-factor points")

# %% [markdown]
# ## 6. The Middle-East airspace cost
# What the reroute penalty actually costs the Asia routes (CPT routes south over Africa, so they
# carry no ME penalty). This is the dollar figure behind the geopolitics headline.

# %%
reroute = (
    econ.with_columns(
        pl.when("me_airspace_exposed")
        .then(REROUTE_HR * BURN_KG_PER_HR / 1000 * JET_USD_PER_TONNE)
        .otherwise(0.0).round(0).alias("reroute_cost_per_flight"),
    )
    .with_columns(
        (pl.col("reroute_cost_per_flight") / CAPACITY).round(1).alias("reroute_cost_per_seat"),
        (pl.col("reroute_cost_per_flight") * pl.col("legs")).round(0).alias("reroute_cost_year"),
    )
)
reroute.filter(pl.col("me_airspace_exposed")).select(
    "route", "legs", "reroute_cost_per_flight", "reroute_cost_per_seat", "reroute_cost_year"
).sort("reroute_cost_year", descending=True)

# %%
print("Annual ME-reroute fuel cost across the exposed routes: "
      f"${reroute['reroute_cost_year'].sum():,.0f}")

# %% [markdown]
# ## 7. Fuel-price sensitivity (the Phase 6 hook)
# Fuel cost per seat as jet fuel swings from a soft $600/t to a spike $1,400/t — per O&D pair. The
# steepest lines (longest sectors) are where a Brent move does the most damage; Phase 6 turns this
# into a full Monte-Carlo margin-at-risk view.

# %%
prices = [600, 800, 1000, 1200, 1400]
pair_fuel = (
    route_ref.group_by("od_pair").agg(pl.col("fuel_tonnes").mean().alias("fuel_tonnes")).sort("od_pair")
)
fig, axes = base.grid(1, ncols=1, figsize=(9, 5.5))
for pair, tonnes in zip(pair_fuel["od_pair"], pair_fuel["fuel_tonnes"]):
    axes[0].plot(prices, [tonnes * p / CAPACITY for p in prices], marker="o", label=pair)
axes[0].axvline(JET_USD_PER_TONNE, ls="--", color="#888", lw=1)
axes[0].set(xlabel="jet fuel ($/tonne)", ylabel="fuel cost per seat ($)",
            title="Fuel cost per seat vs jet-fuel price")
axes[0].legend(title="O&D pair")

# %% [markdown]
# ## 8. Persist the enriched frame
# Attach the route-level fuel economics to every fare row and write the table Phases 4-6 read.

# %%
fuel_cols = ["route_code", "distance_km", "block_hours", "fuel_tonnes", "fuel_cost_usd",
             "fuel_cost_per_seat"]
enriched = clean.join(route_ref.select(fuel_cols), on="route_code", how="left")
problems = validate.check_schema(
    enriched, required=fuel_cols, non_null=["distance_km", "fuel_cost_per_seat"],
    raise_on_error=False,
)
print("schema problems:", problems or "none")
write_parquet(enriched, ENRICHED)
print(f"saved {enriched.height:,} rows x {enriched.width} cols -> {ENRICHED}")

# %% [markdown]
# **Takeaways**
#
# - **Fuel economics are now explicit** per route — distance, block hours, fuel tonnes, fuel cost
#   per flight and per seat — from a documented 787-9 model, no cost column required.
# - **Oil-exposure ranking:** the longest sectors (Cape Town and the Bangkok pairs) carry the
#   highest fuel-cost-per-seat and the largest fuel share of revenue → they hurt first when Brent
#   rises. The `fuel_breakeven_lf` / `cushion_pts` columns say how much buffer each route has.
# - **The Middle-East airspace detour has a price tag** — quantified per flight, per seat, and
#   annualised across the exposed Asia routes (the Cape Town pair is unaffected).
# - **Fare barely tracks distance** (low rank correlation) → pricing is demand/competition-driven,
#   so a fuel shock compresses margin rather than auto-passing through — the case for the active
#   pricing levers in Phases 5-6.
# - **Caveat:** fuel-only economics (the oil-sensitive slice), not full P&L; all engineering
#   constants are labelled assumptions to be replaced with actuals.
#
# **Next (Phase 4):** forecast demand per route over the booking curve — the volume side of the
# revenue equation, with prediction intervals and a rolling-origin backtest.
