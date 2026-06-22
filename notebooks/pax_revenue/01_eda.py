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
# # Pax-revenue · Phase 1 — EDA on the fuel- & airspace-exposed long-hauls
#
# **Decision this informs:** before we model demand or price, we need to know what the raw
# Navitaire fare feed actually contains for the five ultra-long-haul O&D pairs most exposed to
# jet-fuel cost and Middle-East airspace re-routing — **OSL-BKK, OSL-HKT, LGW-CPT, ARN-BKK,
# ARN-HKT** (both directions). These are the routes where an oil spike or an airspace closure
# bites margin first, so they are where revenue actions matter most.
#
# This phase is read-only: it profiles the raw CSV, surfaces the data-quality landmines (refunds,
# **six currencies on one `Charge` column**, the booking-curve confounding), and frames the
# distributions. Phases 2+ consume a cleaned artifact; this one just tells us what we're dealing
# with.
#
# > Source: `data/raw/pax_revenue.csv` (1.18M passenger-segment fares, Jan 2025 → Jan 2026).

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
RAW_CSV = (ROOT / "data" / "raw" / "pax_revenue.csv").as_posix()

# The five O&D pairs, both directions (Route is the concatenated O+D, e.g. "OSLBKK").
ROUTE_CODES = [
    "OSLBKK", "BKKOSL",  # Oslo  <-> Bangkok
    "OSLHKT", "HKTOSL",  # Oslo  <-> Phuket
    "LGWCPT", "CPTLGW",  # London Gatwick <-> Cape Town
    "ARNBKK", "BKKARN",  # Stockholm <-> Bangkok
    "ARNHKT", "HKTARN",  # Stockholm <-> Phuket
]


def pretty(route: str) -> str:
    """`OSLBKK` -> `OSL-BKK`."""
    return f"{route[:3]}-{route[3:]}"


# %% [markdown]
# ## 1. Load
# Two passes over the file, both out-of-core via DuckDB so the 166 MB CSV never lands in pandas:
# one network-wide aggregate to *position* our five pairs against the whole route map, and one
# filtered pull of just the subset rows we will study (~106k rows — small enough to hold eagerly).

# %%
# Network context: where do our five pairs sit in the overall booking volume?
network = query_files(
    f"SELECT Route, count(*) AS n FROM read_csv_auto('{RAW_CSV}', header=true) GROUP BY 1"
)
subset_volume = network.filter(pl.col("Route").is_in(ROUTE_CODES))["n"].sum()
print(f"network: {network['n'].sum():,} pax-segments across {network.height} routes")
print(f"subset : {subset_volume:,} pax-segments ({subset_volume / network['n'].sum():.1%}) "
      f"across {len(ROUTE_CODES)} directional routes")

# %%
in_list = ", ".join(f"'{r}'" for r in ROUTE_CODES)
df = query_files(
    f"SELECT * FROM read_csv_auto('{RAW_CSV}', header=true) WHERE Route IN ({in_list})"
)
# Derived columns we lean on throughout EDA (all promoted to the typed loader in Phase 2).
df = df.with_columns(
    pl.col("Route").map_elements(pretty, return_dtype=pl.String).alias("route"),
    pl.col("Route").str.slice(0, 3).alias("origin"),
    pl.col("Route").str.slice(3, 3).alias("dest"),
    ((pl.col("DepartureTime") - pl.col("BookingUTC")).dt.total_hours() / 24).alias("lead_days"),
)
print(f"loaded {df.height:,} rows x {df.width} cols")
df.head()

# %% [markdown]
# ## 2. Inspect — types, memory, completeness
# `summary` (dtype/nulls/stats per column), `missingness`, `cardinality` (which columns are IDs vs
# categoricals). Know the data before trusting it.

# %%
print(memory_report(df))
df.schema

# %%
stats.summary(df)

# %%
stats.missingness(df)

# %%
stats.cardinality(df)

# %% [markdown]
# ## 3. Data quality — the landmines to clean before modelling
# Three things that will silently corrupt any revenue model if ignored:
# **(a)** non-positive `Charge` rows (refunds / zero fares), **(b)** the **six-currency `Charge`
# column** — averaging it is meaningless until we FX-normalise, **(c)** the booking-curve fields
# (`LFPercentage`, `CumulativePAX`) bounds.

# %%
# (a) Refunds / zero fares — quantify, don't average over them.
neg = df.filter(pl.col("Charge") <= 0).height
print(f"non-positive charges: {neg:,} ({neg / df.height:.2%}) — treated separately, not as fares")
stats.describe_distribution(df.filter(pl.col("Charge") > 0)["Charge"].to_numpy())

# %%
# (b) The currency trap: the SAME long-haul seat shows up as ~280 (USD) or ~3,500 (NOK).
# This is why every cross-route number in later phases runs on an FX-normalised column.
by_ccy = (
    df.group_by("CurrencyCode")
    .agg(
        pl.len().alias("n"),
        pl.col("Charge").filter(pl.col("Charge") > 0).mean().round(0).alias("avg_charge_local"),
    )
    .sort("n", descending=True)
)
by_ccy

# %%
# Currency mix per route — confirms each O&D is dominated by its home-market currency
# (NOK/SEK out of Scandinavia, GBP/ZAR on the Cape Town pair, USD almost everywhere).
ccy_by_route = (
    df.group_by("route", "CurrencyCode")
    .agg(pl.len().alias("n"))
    .sort(["route", "n"], descending=[False, True])
)
ccy_by_route

# %%
# (c) Schema gate + business rules — fail-fast contract for downstream phases.
problems = validate.check_schema(
    df,
    required=["BookingID", "PassengerID", "SegmentID", "Route", "Charge", "CurrencyCode",
              "BookingUTC", "DepartureTime", "LFPercentage"],
    non_null=["Route", "Charge", "CurrencyCode", "DepartureTime"],
    raise_on_error=False,
)
print("schema problems:", problems or "none")

rules = validate.check_rules(
    df,
    {
        "load factor in [0, 110]": pl.col("LFPercentage").is_between(0, 110),
        "departs after booking": pl.col("DepartureTime") >= pl.col("BookingUTC"),
        "cumulative pax <= capacity": pl.col("CumulativePAX") <= pl.col("FlCapacity"),
        "lead time under 400 days": pl.col("lead_days") <= 400,
    },
)
rules

# %% [markdown]
# ## 4. Revenue & volume landscape
# Volume per route (pax-segments), the booking entities behind it, and how it spreads over the
# year. Revenue itself waits for FX normalisation (Phase 2) — here we count seats sold.

# %%
route_vol = (
    df.group_by("route")
    .agg(
        pl.len().alias("pax_segments"),
        pl.col("BookingID").n_unique().alias("bookings"),
        pl.col("InventoryLegID").n_unique().alias("legs"),
        pl.col("Charge").filter(pl.col("Charge") > 0).count().alias("paid_fares"),
    )
    .sort("pax_segments", descending=True)
)
route_vol

# %%
fig, axes = base.grid(2)
eda.count_bar(df, "route", ax=axes[0], title="Pax-segments by route")
eda.count_bar(df, "CurrencyCode", ax=axes[1], title="Pax-segments by currency")

# %%
# Monthly booking activity vs monthly departures — when revenue is *taken* vs when it is *flown*.
monthly = (
    df.select(
        pl.col("BookingUTC").dt.truncate("1mo").alias("month"),
    )
    .group_by("month").agg(pl.len().alias("bookings_made")).sort("month")
    .join(
        df.select(pl.col("DepartureTime").dt.truncate("1mo").alias("month"))
        .group_by("month").agg(pl.len().alias("pax_departing")),
        on="month", how="full", coalesce=True,
    )
    .sort("month")
)
monthly

# %% [markdown]
# ## 5. Fare distribution & shape
# Right-skewed positive amounts almost always read as log-normal/gamma — which decides whether a
# linear model needs a log transform later. We fit on the **single dominant currency** so we are
# describing one real fare scale, not a currency artefact.

# %%
top_ccy = by_ccy["CurrencyCode"][0]
fares = df.filter((pl.col("CurrencyCode") == top_ccy) & (pl.col("Charge") > 0))["Charge"].to_numpy()
print(f"fitting fare distribution on {fares.size:,} {top_ccy} fares")
stats.best_distribution(fares, candidates=["lognorm", "gamma", "weibull_min", "expon", "norm"])

# %%
winner = stats.best_distribution(fares, candidates=["lognorm", "gamma", "weibull_min", "norm"])["dist"][0]
params = stats.fit_distribution(fares, winner)["params"]
print(f"best fit: {winner}; raw skew {stats.describe_distribution(fares)['skew']:+.2f}")

fig, axes = base.grid(3, ncols=3)
eda.histogram(fares, ax=axes[0], title=f"Fare ({top_ccy})")
eda.ecdf(fares, ax=axes[1], title=f"Fare ECDF ({top_ccy})")
eda.fit_overlay(fares, dist=winner, params=params, ax=axes[2], title=f"Fare vs {winner} fit")

# %%
# Fare by route, held to one currency so the boxes are comparable. Cape Town and the Bangkok
# long-hauls carry the heaviest fares — exactly the routes most fuel-exposed.
fig, axes = base.grid(1, ncols=1, figsize=(11, 5))
eda.boxplot_by(
    df.filter((pl.col("CurrencyCode") == top_ccy) & (pl.col("Charge") > 0)),
    "Charge", "route", ax=axes[0], title=f"Fare by route ({top_ccy})",
)

# %% [markdown]
# ## 6. The booking curve — and the confounding it creates
# Each row records `LFPercentage` (how full the flight was) and `lead_days` (days to departure) at
# the moment of booking. Fares **rise as the flight fills and as departure nears** — which means a
# naive demand curve fit on this data slopes the *wrong* way (price and quantity move together).
# Any elasticity in Phase 5 must condition on fill stage; this cell is the warning.

# %%
fig, axes = base.grid(3, ncols=3)
eda.histogram(df["lead_days"].to_numpy(), ax=axes[0], title="Lead time (days to departure)")
eda.histogram(df["LFPercentage"].to_numpy(), ax=axes[1], title="Load factor at booking (%)")
eda.scatter(
    df.filter((pl.col("CurrencyCode") == top_ccy) & (pl.col("Charge") > 0))
    .pipe(transform.sample, n=4000, seed=42),
    "LFPercentage", "Charge", ax=axes[2], title=f"Fare vs load factor ({top_ccy})",
)

# %%
# Mean fare by load-factor decile (one currency): the realized fare ladder airlines price along.
ladder = (
    df.filter((pl.col("CurrencyCode") == top_ccy) & (pl.col("Charge") > 0))
    .with_columns((pl.col("LFPercentage") // 10 * 10).cast(pl.Int32).alias("lf_band"))
    .group_by("lf_band")
    .agg(pl.col("Charge").mean().round(0).alias("avg_fare"), pl.len().alias("n"))
    .sort("lf_band")
)
ladder

# %% [markdown]
# ## 7. Seasonality
# Long-haul leisure to Asia/Africa is strongly seasonal — the booking and departure profile tells
# us what a forecast (Phase 4) must capture.

# %%
fig, axes = base.grid(2)
eda.boxplot_by(
    df.with_columns(pl.col("DepartureTime").dt.month().cast(pl.String).alias("dep_month")),
    "lead_days", "dep_month", ax=axes[0], title="Lead time by departure month",
)
eda.count_bar(
    df.with_columns(pl.col("DepartureTime").dt.strftime("%a").alias("dep_weekday")),
    "dep_weekday", ax=axes[1], title="Departures by weekday",
)

# %% [markdown]
# ## 8. Fare-basis structure (preview of Phase 2 feature engineering)
# `FareBasis` is the pricing DNA — it encodes booking class and fare brand. We don't parse it
# properly here, just confirm the structure: a leading **booking-class letter** and a trailing
# **brand code** (EL/EP …). Phase 2 turns this into clean `booking_class` / `fare_brand` columns.

# %%
print(f"distinct fare-basis codes: {df['FareBasis'].n_unique()}")
fb = df.with_columns(
    pl.col("FareBasis").str.slice(0, 1).alias("class_letter"),
    pl.col("FareBasis").str.slice(-2).alias("brand_tail"),
)

# %%
# Leading booking-class letter (Y/Q/N/V/O/L/H… — the demand bucket).
fb["class_letter"].value_counts(sort=True).head(8)

# %%
# Trailing brand code (EL/EP… — the fare family / cabin brand).
fb["brand_tail"].value_counts(sort=True).head(8)

# %% [markdown]
# ## 9. Numeric relationships
# Spearman (rank, robust) across the booking-curve numerics. The positive `LFPercentage` ↔ `Charge`
# link is the confounding from §6 in one number — not a demand relationship.

# %%
stats.spearman(df.select(["Charge", "LFPercentage", "CumulativePAX", "lead_days", "FlCapacity"]))

# %%
fig, axes = base.grid(1, ncols=1, figsize=(7, 6))
eda.correlation_heatmap(
    df.select(["Charge", "LFPercentage", "CumulativePAX", "lead_days", "FlCapacity"]),
    ax=axes[0], title="Booking-curve correlations (Pearson)",
)

# %% [markdown]
# **Takeaways**
#
# - **Scope confirmed:** the five pairs are ~9% of network volume (106k pax-segments) and skew to
#   the heaviest fares — the right place to defend margin against fuel/airspace shocks.
# - **Must FX-normalise (Phase 2):** `Charge` mixes USD/GBP/EUR/NOK/SEK/ZAR; raw averages are
#   meaningless. Each route is dominated by its home currency.
# - **Refunds/zero fares (~small %)** must be segregated, not averaged into fares.
# - **Fares are right-skewed** (log-normal/gamma) → linear models will want a log transform.
# - **Booking-curve confounding is real:** fare rises with load factor and toward departure, so
#   Phase 5 elasticity must condition on fill stage (and the honest causal version needs IV /
#   experimental price variation, which this observational feed can't supply).
# - **Seasonality + fare-basis structure** are strong and parseable → Phase 2 feature engineering,
#   Phase 4 forecasting.
#
# **Next (Phase 2):** typed cached loader → FX-normalise to USD → parse `FareBasis` → engineer
# lead-time / booking-curve / seasonality features → write the clean Parquet the rest consume.
