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
# # Pax-revenue · Phase 5 — Price response & where fare has room to move
#
# **Decision this informs:** on which routes can we lift fare to defend margin against fuel, and
# where would that just shed volume? Three layers, in order of how much we can trust them:
#
# 1. **The RM fare ladder (descriptive, reliable):** how realized fare actually moves with fill and
#    booking window — the pricing behaviour already being executed.
# 2. **Observational elasticity (directional, confounded):** a leg-level demand slope per route —
#    with the loud caveat that revenue-management *sets* price in response to demand, so the sign is
#    biased and a causal read needs price experiments / IV.
# 3. **Where fare can move:** margin cushion (Phase 3) × elasticity sign, plus an optimal-price
#    sensitivity that shows the answer hinges on the (unknown) elasticity.
#
# > Input: `data/processed/pax_revenue_enriched.parquet`

# %%
from core.config import ROOT
from core.prelude import *
from core.pricing import elasticity as elast
from core.pricing import optimize as popt

set_theme()
ENRICHED = ROOT / "data" / "processed" / "pax_revenue_enriched.parquet"

fares = read_parquet(ENRICHED).filter(~pl.col("is_refund"))
print(f"{fares.height:,} paid fares across {fares['route'].n_unique()} routes")

# %% [markdown]
# ## 1. The RM fare ladder (descriptive — the part we can trust)
# Mean realized fare by **fill stage** (load-factor band) and by **booking window** (lead-time
# band), split by market. Fare climbs as the cabin fills and as departure nears — the revenue
# manager's ladder, read straight off accepted fares.

# %%
LF_ORDER = list(range(0, 110, 10))
LEAD_ORDER = ["<2w", "2-4w", "1-2m", "2-3m", "3-6m", "6m+"]

ladder_lf = (
    fares.with_columns((pl.col("lf_at_booking") // 10 * 10).cast(pl.Int32).alias("lf_band"))
    .group_by("lf_band", "market").agg(pl.col("charge_usd").mean().alias("avg_fare"))
    .sort("lf_band")
)
ladder_lead = (
    fares.group_by("lead_days_bin", "market").agg(pl.col("charge_usd").mean().alias("avg_fare"))
)

fig, axes = base.grid(2)
for mkt in ladder_lf["market"].unique().sort():
    d = ladder_lf.filter(pl.col("market") == mkt).sort("lf_band")
    axes[0].plot(d["lf_band"], d["avg_fare"], marker="o", label=mkt)
axes[0].set(xlabel="load factor at booking (%)", ylabel="avg fare (USD)",
            title="Fare ladder vs fill")
axes[0].legend()
for mkt in ladder_lead["market"].unique().sort():
    d = ladder_lead.filter(pl.col("market") == mkt)
    d = d.with_columns(pl.col("lead_days_bin").cast(pl.String)).sort(
        pl.col("lead_days_bin").map_elements(lambda b: LEAD_ORDER.index(b) if b in LEAD_ORDER else 99,
                                             return_dtype=pl.Int64))
    axes[1].plot(d["lead_days_bin"], d["avg_fare"], marker="o", label=mkt)
axes[1].set(xlabel="booking window", ylabel="avg fare (USD)", title="Fare ladder vs lead time")
axes[1].legend()

# %% [markdown]
# ### Quantify the ladder with OLS (read the effects, with assumptions checked)
# `charge_usd ~ load factor + lead days + direction + market`. The coefficients say how many dollars
# the fare moves per fill-point and per lead-day. We check residual assumptions before trusting the
# p-values — fares are skewed, so we expect heteroscedasticity and read the SEs as approximate.

# %%
reg = (
    fares.with_columns(
        (pl.col("direction") == "outbound").cast(pl.Int8).alias("outbound"),
        (pl.col("market") == "SE Asia").cast(pl.Int8).alias("is_asia"),
    )
    .select("charge_usd", "lf_at_booking", "lead_days", "outbound", "is_asia")
    .pipe(transform.sample, n=5000, seed=42)
)
X = ["lf_at_booking", "lead_days", "outbound", "is_asia"]
fit = regression.ols_fit(reg, y="charge_usd", x=X)
fit.coefficients

# %%
design = np.column_stack([np.ones(reg.height)] + [reg[c].to_numpy() for c in X])
resid = reg["charge_usd"].to_numpy() - design @ fit.coefficients["coef"].to_numpy()
print(f"R^2 = {fit.r_squared:.3f}  (fare is set by far more than these four levers)")
print("assumptions:", regression.linear_assumptions(reg.select(X), resid))

# %% [markdown]
# ## 2. Observational elasticity per route — directional, and confounded
# At the **leg** level: each flight's average fare vs the passengers it carried, fit log-log per
# route (`pricing.elasticity.segment_elasticity`).
#
# > **Identification caveat (the one that decides trust).** These prices were *set in response to
# > expected demand* (high-demand legs are priced up **and** fill up), and peak-season legs are both
# > dearer and fuller. That biases the slope upward — a positive or "inelastic" estimate here is the
# > expected artefact of observational RM data, **not** a causal demand elasticity. The playbook gate:
# > the CI must exclude **−1** before any raise/cut call. The clean read needs price experiments or an
# > instrument (`causal.iv_effect`). The Phuket pairs have too few legs (~5) to fit and drop out.
# > **Watch for positive estimates below** — a positive slope is impossible for real demand, so it is
# > the confounding caught red-handed, not a finding.

# %%
leg = (
    fares.group_by("leg_id").agg(
        pl.col("route").first(),
        pl.col("charge_usd").mean().alias("avg_fare"),
        pl.len().alias("pax"),
    )
)
elastic = (
    elast.segment_elasticity(leg, price="avg_fare", quantity="pax", segment="route", min_rows=20)
    .rename({"segment": "route"})
    .with_columns(
        (pl.col("ci_high") < -1).alias("elastic"),
        (pl.col("ci_low") > -1).alias("inelastic"),
        ((pl.col("ci_low") <= -1) & (pl.col("ci_high") >= -1)).alias("ci_spans_-1"),
    )
)
elastic.select("route", "n", "elasticity", "ci_low", "ci_high", "elastic", "inelastic",
               "ci_spans_-1").sort("elasticity")

# %%
# Is one elasticity even the right model? Curvature check on the busiest route's legs.
busiest = leg.group_by("route").agg(pl.len().alias("legs")).sort("legs", descending=True)["route"][0]
bl = leg.filter(pl.col("route") == busiest)
print(f"nonlinearity check on {busiest} ({bl.height} legs):")
print(elast.nonlinear_elasticity_check(bl["avg_fare"].to_numpy(), bl["pax"].to_numpy()))

# %% [markdown]
# ## 3. Where can fare move? Margin cushion × price response
# Pair each route's fuel exposure (Phase 3) with what we (cautiously) learned about price response.
# In practice **no route's elasticity is cleanly identified** here — two even show physically-
# impossible *positive* slopes (pure confounding) — so the table resolves to a **price-test plan
# prioritised by fuel exposure** (`fuel_to_fare`), not a list of fares to raise. The "room to raise"
# branch only fires on a genuinely-negative, inelastic, CI-clean estimate (none qualify today).

# %%
route_econ = fares.group_by("route").agg(
    pl.col("charge_usd").mean().round(0).alias("avg_fare"),
    pl.col("fuel_cost_per_seat").first().round(0).alias("fuel_per_seat"),
)
opportunity = (
    route_econ.join(elastic.select("route", "elasticity", "ci_low", "ci_high",
                                    "inelastic", "ci_spans_-1"), on="route", how="left")
    .with_columns((pl.col("fuel_per_seat") / pl.col("avg_fare")).round(2).alias("fuel_to_fare"))
    .with_columns(
        pl.when(pl.col("elasticity").is_null()).then(pl.lit("test (too few legs)"))
        .when(pl.col("elasticity") >= 0).then(pl.lit("confounded (e>=0) -> price test"))
        .when(pl.col("inelastic") & (pl.col("elasticity") < 0) & (pl.col("fuel_to_fare") > 0.35))
        .then(pl.lit("room to raise (verify w/ test)"))
        .when(pl.col("ci_spans_-1")).then(pl.lit("not identified -> price test"))
        .when(pl.col("ci_high") < -1).then(pl.lit("demand elastic -> hold/compete"))
        .otherwise(pl.lit("price test")).alias("action")
    )
    .sort("fuel_to_fare", descending=True)
)
opportunity

# %% [markdown]
# ### Optimal price hinges on the (unknown) elasticity — so show the whole curve
# For the thinnest-cushion route, sweep plausible elasticities through the constant-elasticity
# optimum (`markup_price`, fuel-only unit cost) and compare to today's fare. The spread is the point:
# without a price test the "right" fare ranges widely — which is itself the recommendation.

# %%
TARGET = "CPT-LGW"  # thinnest fuel cushion from Phase 3
sub = fares.filter(pl.col("route") == TARGET)
ucost = float(sub["fuel_cost_per_seat"][0])
p0 = float(sub["charge_usd"].mean())
q0 = float(leg.filter(pl.col("route") == TARGET)["pax"].mean())
print(f"{TARGET}: current avg fare ${p0:,.0f} | fuel cost/seat ${ucost:,.0f}")

es = np.linspace(-2.6, -1.15, 30)
markup = [popt.markup_price(e, ucost) for e in es]
# Sign of marginal profit at the current fare: MR = p*(1 + 1/e) rises with p and is 0 at the
# optimum, so mp < 0 means price is *below* the optimum -> raise; mp > 0 -> already past it -> cut.
for e in (-1.3, -1.6, -2.2):
    mp = float(np.asarray(popt.marginal_profit(e, [p0], unit_cost=ucost))[0])
    print(f"  e={e}: marginal profit at current fare = {mp:+.1f}  -> {'raise' if mp < 0 else 'cut'}")

fig, axes = base.grid(2)
axes[0].plot(-es, markup, color="tab:blue")
axes[0].axhline(p0, ls="--", color="tab:red", label=f"current ${p0:,.0f}")
axes[0].set(xlabel="|elasticity|", ylabel="profit-max fare (USD)",
            title=f"{TARGET}: optimal fare vs assumed elasticity")
axes[0].legend()

# Illustrative price curve at one plausible elasticity, anchored on observed (fare, pax).
e_show = -1.5
intercept = np.log(q0) - e_show * np.log(p0)
grid = np.linspace(p0 * 0.5, p0 * 1.8, 60)
opt_price, _ = popt.optimal_price(intercept, e_show, grid, unit_cost=ucost)
sched = pl.DataFrame({
    "price": grid,
    "revenue": popt.revenue_at(intercept, e_show, grid),
    "profit": popt.profit_at(intercept, e_show, grid, unit_cost=ucost),
})
business.price_curves(sched, price="price", curves=("revenue", "profit"), optimum=opt_price,
                      ax=axes[1], title=f"{TARGET}: revenue/profit vs price (illustrative, e={e_show})")

# %% [markdown]
# **Takeaways**
#
# - **The fare ladder is real and reliable:** fare rises with fill and toward departure (OLS
#   coefficients in §1) — the executed RM behaviour, read straight off accepted fares.
# - **Elasticity is *not* causally identified here.** The leg-level slopes are confounded by RM
#   reacting to demand and by seasonality; treat the per-route numbers as directional and check the
#   CI against −1. Phuket pairs can't be fit at all (too few legs). The honest next step for a real
#   pricing decision is a **price experiment** (or an instrument) — exactly what `analytics.causal`
#   exists for.
# - **Where fare can move — honestly:** *no* route's elasticity is causally identified here (signs
#   are confounded; BKK-ARN and ARN-BKK even read *positive*). So §3 is a **price-test plan
#   prioritised by fuel exposure**, not a raise list, and §3.5 shows the optimal fare for the
#   thinnest-cushion route (CPT-LGW) swings widely with the assumed elasticity — pinned down only
#   once a test delivers a credible number. The reliable levers today are the **descriptive ladder**
#   and the **fuel-exposure ranking**.
# - **This is the volume×price half of the margin story; Phase 6 closes it** by pushing demand
#   (Phase 4) and these price levers through an oil-price × FX × airspace Monte-Carlo to rank which
#   routes actually go underwater and how big a fare move is needed to hold target margin.
