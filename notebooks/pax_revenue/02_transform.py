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
# # Pax-revenue · Phase 2 — Transform → the clean, FX-normalised, feature-rich frame
#
# **Decision this informs:** every later phase (forecast, price, oil/FX scenario) needs one
# trustworthy fare-level table — money on a single scale, refunds separated, the booking curve and
# calendar turned into features, and the geography labelled. This phase builds that table once and
# writes it to `data/processed/` so Phases 3-6 just read it.
#
# What it demonstrates about the repo: the **typed source catalog** (a schema pin), the `@cached`
# loader (expensive pull → Parquet in `data/interim`), and the stateless `features.*` transforms
# (`temporal`, `transform`, `clean`, `validate`).
#
# > Input: `data/raw/pax_revenue.csv` · Output: `data/processed/pax_revenue_clean.parquet`

# %%
from core.config import ROOT
from core.prelude import *

set_theme()
RAW_CSV = (ROOT / "data" / "raw" / "pax_revenue.csv").as_posix()
CLEAN_PARQUET = ROOT / "data" / "processed" / "pax_revenue_clean.parquet"

ROUTE_CODES = [
    "OSLBKK", "BKKOSL", "OSLHKT", "HKTOSL",
    "LGWCPT", "CPTLGW", "ARNBKK", "BKKARN", "ARNHKT", "HKTARN",
]
HUBS = {"OSL", "ARN", "LGW"}  # European home bases; the other end is the leisure market

# %% [markdown]
# ## 1. Register the source (schema pin) and load the subset (cached)
# `catalog.register` records *where* the data lives and the dtypes it must satisfy — the
# "one typed loader per source" rule. We then pull only the five-pair subset through a `@cached`
# DuckDB loader, so the 166 MB CSV is scanned once and the result is reused from Parquet.

# %%
catalog.register(
    "pax_revenue",
    RAW_CSV,
    format="csv",
    schema={
        "BookingID": pl.Int64, "PassengerID": pl.Int64, "SegmentID": pl.Int64,
        "InventoryLegID": pl.Int64, "Route": pl.String, "CurrencyCode": pl.String,
        "Charge": pl.Float64, "LFPercentage": pl.Float64, "CumulativePAX": pl.Int64,
        "FlCapacity": pl.Int64, "FareBasis": pl.String,
    },
)
print("registered sources:", catalog.sources())
catalog.describe("pax_revenue")


# %%
@cached
def load_pax_subset(routes: tuple[str, ...]) -> pl.DataFrame:
    """Scan the raw CSV out-of-core and keep only the requested routes (cached to data/interim)."""
    in_list = ", ".join(f"'{r}'" for r in routes)
    return query_files(
        f"SELECT * FROM read_csv_auto('{RAW_CSV}', header=true) WHERE Route IN ({in_list})"
    )


raw = load_pax_subset(tuple(ROUTE_CODES))
print(f"raw subset: {raw.height:,} rows x {raw.width} cols")
raw.head()

# %% [markdown]
# ## 2. Rename to clean snake_case
# The feed ships PascalCase; the clean artifact is snake_case so downstream code reads naturally.

# %%
RENAME = {
    "BookingUTC": "booking_utc", "BookingID": "booking_id", "PassengerID": "passenger_id",
    "SegmentID": "segment_id", "RecordLocator": "record_locator", "InventoryLegID": "leg_id",
    "DepartureTime": "departure_time", "ArrivalTime": "arrival_time", "FlightNumber": "flight_number",
    "Route": "route_code", "CumulativePAX": "cumulative_pax", "FlCapacity": "capacity",
    "LFPercentage": "lf_at_booking", "FareBasis": "fare_basis", "CurrencyCode": "currency",
    "Charge": "charge_local",
}
df = raw.rename(RENAME)
df.columns

# %% [markdown]
# ## 3. FX-normalise `charge_local` → `charge_usd`
# The single most important fix from Phase 1: six currencies on one column. We convert to USD with
# **documented period-average 2025 rates** (USD per 1 unit of local currency). These are
# placeholders structured to be swapped for a daily FX feed keyed on `booking_utc` — NOK and ZAR in
# particular move with oil/risk sentiment, which is exactly the Phase 6 story.

# %%
# USD per 1 unit of local currency (period-average 2025; replace with a dated FX feed in prod).
FX_TO_USD = {"USD": 1.0, "GBP": 1.28, "EUR": 1.08, "NOK": 0.094, "SEK": 0.095, "ZAR": 0.054}

df = df.with_columns(
    pl.col("currency").replace_strict(FX_TO_USD, return_dtype=pl.Float64).alias("fx_to_usd"),
).with_columns(
    (pl.col("charge_local") * pl.col("fx_to_usd")).round(2).alias("charge_usd"),
)

# Proof it worked: local averages span 240→6,000; in USD the same long-haul seat lands on one scale.
fx_check = (
    df.filter(pl.col("charge_local") > 0)
    .group_by("currency")
    .agg(
        pl.len().alias("n"),
        pl.col("charge_local").mean().round(0).alias("avg_local"),
        pl.col("charge_usd").mean().round(0).alias("avg_usd"),
    )
    .sort("n", descending=True)
)
fx_check

# %% [markdown]
# ## 4. Segregate refunds
# Non-positive `Charge` rows are refunds / zero fares, not demand. We flag them so fare analysis
# uses paid fares only, while the net-revenue view (Phase 6) can still subtract them.

# %%
df = df.with_columns((pl.col("charge_local") <= 0).alias("is_refund"))
print(df["is_refund"].value_counts(sort=True))

# %% [markdown]
# ## 5. Parse `fare_basis` → booking class + fare brand
# The pricing DNA: a leading **booking-class letter** (the demand bucket) and a trailing
# **brand code** (EL/EP… the fare family). Mapping brand codes to marketing names needs the fare
# catalog; here we keep the codes as clean categorical features.

# %%
df = df.with_columns(
    pl.col("fare_basis").str.slice(0, 1).alias("booking_class"),
    pl.col("fare_basis").str.extract(r"([A-Z]{2})$", 1).fill_null("other").alias("fare_brand"),
)
df.group_by("booking_class").agg(pl.len().alias("n")).sort("n", descending=True).head(10)

# %% [markdown]
# ## 6. Route geography & market labels
# Origin/dest, undirected O&D pair, travel direction (out of the European hub vs back), the leisure
# market, and the two exposure flags that drive Phase 6: **fuel-exposed** (all five are
# ultra-long-haul) and **Middle-East-airspace-exposed** (the Europe↔Asia pairs that overfly the ME
# corridor; the Cape Town pair routes south over Africa instead).

# %%
REGION = {"OSL": "N. Europe", "ARN": "N. Europe", "LGW": "N. Europe",
          "BKK": "SE Asia", "HKT": "SE Asia", "CPT": "S. Africa"}

df = df.with_columns(
    pl.col("route_code").str.slice(0, 3).alias("origin"),
    pl.col("route_code").str.slice(3, 3).alias("dest"),
).with_columns(
    (pl.col("origin").str.slice(0, 3) + "-" + pl.col("dest")).alias("route"),
    pl.when(pl.col("origin").is_in(HUBS)).then(pl.lit("outbound")).otherwise(pl.lit("inbound")).alias("direction"),
    # leisure airport = the non-hub end; its region is the market we sell into
    pl.when(pl.col("origin").is_in(HUBS)).then(pl.col("dest")).otherwise(pl.col("origin")).alias("leisure_airport"),
).with_columns(
    pl.col("leisure_airport").replace_strict(REGION, return_dtype=pl.String).alias("market"),
    # undirected pair label, e.g. both OSLBKK and BKKOSL -> "BKK-OSL"
    pl.concat_list(["origin", "dest"]).list.sort().list.join("-").alias("od_pair"),
).with_columns(
    pl.col("leisure_airport").is_in(["BKK", "HKT"]).alias("me_airspace_exposed"),
    pl.lit(True).alias("fuel_exposed"),
)
df.group_by("od_pair", "direction").agg(pl.len().alias("n")).sort("od_pair")

# %% [markdown]
# ## 7. Temporal & booking-curve features
# Lead time and its bands, booking/departure calendar parts, cyclical month encoding (so Dec sits
# next to Jan), the meteorological season, and the per-leg booking-curve position: how full the
# flight ends up (`leg_final_lf`) and how much fill is still to come when this passenger books
# (`fill_remaining`). These are the features Phases 4-5 model on.

# %%
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}

df = df.with_columns(
    ((pl.col("departure_time") - pl.col("booking_utc")).dt.total_hours() / 24).round(1).alias("lead_days"),
    pl.col("booking_utc").dt.month().alias("booking_month"),
    pl.col("booking_utc").dt.weekday().alias("booking_dow"),
    pl.col("departure_time").dt.month().alias("dep_month"),
    pl.col("departure_time").dt.weekday().alias("dep_dow"),
    pl.col("departure_time").dt.date().alias("dep_date"),
).with_columns(
    pl.col("dep_month").replace_strict(SEASON, return_dtype=pl.String).alias("dep_season"),
)

# Lead-time bands (booking-window segments) and cyclical month encoding via core helpers.
df = transform.discretize(
    df, "lead_days",
    breaks=[14, 30, 60, 90, 180],
    labels=["<2w", "2-4w", "1-2m", "2-3m", "3-6m", "6m+"],
)
df = temporal.cyclical_encode(df, "dep_month", period=12)

# Per-leg booking curve: final fill of the leg, and remaining fill at this booking moment.
leg_final = df.group_by("leg_id").agg(pl.col("lf_at_booking").max().alias("leg_final_lf"))
df = df.join(leg_final, on="leg_id", how="left").with_columns(
    (pl.col("leg_final_lf") - pl.col("lf_at_booking")).alias("fill_remaining"),
)
df.select(["lead_days", "lead_days_bin", "dep_season", "lf_at_booking", "leg_final_lf",
           "fill_remaining"]).head()

# %% [markdown]
# ## 8. Validate the transformed contract, then shrink and write
# Fail fast if the new columns are wrong, downcast for memory, and persist to Parquet.

# %%
problems = validate.check_schema(
    df,
    required=["charge_usd", "currency", "route", "booking_class", "lead_days", "dep_season",
              "direction", "market", "leg_final_lf"],
    non_null=["charge_usd", "route", "booking_class", "lead_days"],
    raise_on_error=False,
)
print("schema problems:", problems or "none")

validate.check_rules(
    df,
    {
        "fares have positive USD": pl.col("is_refund") | (pl.col("charge_usd") > 0),
        "fill_remaining non-negative": pl.col("fill_remaining") >= 0,
        "lead_days non-negative": pl.col("lead_days") >= 0,
        "booking_class is a letter": pl.col("booking_class").str.contains(r"^[A-Z]$"),
    },
)

# %%
# Downcast numerics for memory, but keep text as String (not Categorical) so downstream `.str`
# ops and joins behave predictably across phases.
compact = clean.downcast(df).with_columns(cs.categorical().cast(pl.String))
print(memory_report(df))
print("\n" + memory_report(compact))

# %%
write_parquet(compact, CLEAN_PARQUET)
print(f"saved {compact.height:,} rows x {compact.width} cols -> {CLEAN_PARQUET}")

# %% [markdown]
# ## 9. Immediate analytical payoff of the clean frame
# With money on one scale we can finally read **real revenue** by route, direction, and fare brand
# — the first numbers in the whole project that are actually comparable across markets.

# %%
fares = compact.filter(~pl.col("is_refund"))
rev_by_route = (
    fares.group_by("route")
    .agg(
        pl.len().alias("pax_segments"),
        pl.col("charge_usd").sum().round(0).alias("revenue_usd"),
        pl.col("charge_usd").mean().round(0).alias("avg_fare_usd"),
    )
    .sort("revenue_usd", descending=True)
)
rev_by_route

# %%
fig, axes = base.grid(3, ncols=3)
eda.boxplot_by(fares, "charge_usd", "od_pair", ax=axes[0], title="Fare (USD) by O&D pair")
eda.boxplot_by(fares, "charge_usd", "direction", ax=axes[1], title="Fare (USD) by direction")
eda.boxplot_by(
    fares.filter(pl.col("fare_brand").is_in(
        fares["fare_brand"].value_counts(sort=True)["fare_brand"].head(4).to_list()
    )),
    "charge_usd", "fare_brand", ax=axes[2], title="Fare (USD) by top fare brand",
)

# %%
# Revenue split outbound vs inbound, per market — a first look at directional asymmetry.
(
    fares.group_by("market", "direction")
    .agg(pl.col("charge_usd").sum().round(0).alias("revenue_usd"), pl.len().alias("n"))
    .sort(["market", "direction"])
)

# %% [markdown]
# **Takeaways**
#
# - **One clean table written** to `data/processed/pax_revenue_clean.parquet` — FX-normalised,
#   refunds flagged, fare basis parsed, booking-curve + calendar + geography features attached.
#   Phases 3-6 read this and nothing else.
# - **FX normalisation unlocks real comparison:** local averages spanning 240→6,000 collapse onto a
#   common USD scale, so route/market revenue is finally meaningful.
# - **Feature set ready:** `lead_days`(+bands), `dep_season`, cyclical month, `booking_class`,
#   `fare_brand`, `leg_final_lf`, `fill_remaining`, `direction`, `market`, and the
#   `fuel_exposed` / `me_airspace_exposed` flags that Phase 6 prices oil & airspace risk against.
# - **Caveat:** FX rates are documented period-average placeholders; wire a dated feed before any
#   hard financial call (NOK/ZAR co-move with oil — material for the Phase 6 scenarios).
#
# **Next (Phase 3):** enrich each route with great-circle distance → fuel-burn & cost-per-seat,
# the Middle-East reroute penalty, and the break-even load factor that turns oil price into a
# revenue requirement.
