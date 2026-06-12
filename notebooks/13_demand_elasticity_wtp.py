# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 13 · Demand & elasticity — uncertainty, segments, dynamics, and willingness-to-pay
#
# Notebook 12 estimated *one* elasticity and priced off it. This notebook is everything a real
# pricing decision needs around that number: how *sure* we are (CIs, bootstrap), who exactly is
# price-sensitive (cross-price, segments), whether the number is even stable (rolling fits, drift
# test, nonlinearity), why the portfolio elasticity moved (decomposition), and what customers
# would actually pay (purchase-probability WTP, Van Westendorp). Synthetic data with known truth
# throughout, so every estimator is checked against the answer.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. The elasticity *with its uncertainty*
# 60 weeks of randomized price tests (true elasticity **-1.8**). The point estimate alone invites
# overconfidence — the CI is what tells us whether the raise-vs-cut call is identified at all:
# a CI spanning -1 would mean we can't even tell elastic from inelastic.

# %%
price = rng.uniform(60.0, 140.0, 420)
quantity = np.exp(8.0) * price**-1.8 * rng.lognormal(0.0, 0.18, 420)

fit = pricing.elasticity.fit_demand_ci(price, quantity)
boot_low, boot_high = pricing.elasticity.bootstrap_elasticity(price, quantity, n_boot=1000)
print(f"elasticity {fit.elasticity:.2f}  (t-CI {fit.ci_low:.2f} … {fit.ci_high:.2f}, n={fit.n})")
print(f"bootstrap CI         ({boot_low:.2f} … {boot_high:.2f}) — agreement = robustness check")
print(f"CI excludes -1: {fit.ci_high < -1.0} -> the 'elastic' verdict is identified")

# %% [markdown]
# ## 2. Cross-price elasticity — substitutes and complements
# Our demand also responds to a rival's price (true cross-elasticity **+0.9**, a substitute) and
# to an add-on service we bundle (true **-0.5**, a complement). Estimating jointly matters:
# competitor prices co-move with ours, and a univariate fit would absorb their effect into our
# own-price slope (omitted-variable bias).

# %%
rival = rng.uniform(50.0, 150.0, 420)
addon = rng.uniform(5.0, 25.0, 420)
quantity_multi = (
    np.exp(7.0) * price**-1.8 * rival**0.9 * addon**-0.5 * rng.lognormal(0.0, 0.15, 420)
)
pricing.elasticity.cross_price_elasticity(
    quantity_multi, price, {"rival_fare": rival, "addon_service": addon}
)

# %% [markdown]
# ## 3. Segment-level elasticity — who actually reacts to price
# Business customers (true e = **-0.9**, inelastic) vs leisure (true **-2.6**, highly elastic).
# One pooled number would mis-price both. The differentiated-pricing read: protect margin where
# |e| < 1, compete on price where |e| is large.

# %%
segments = []
for name, e_true, n in [("business", -0.9, 300), ("leisure", -2.6, 500)]:
    p_seg = rng.uniform(60.0, 140.0, n)
    q_seg = np.exp(8.0) * p_seg**e_true * rng.lognormal(0.0, 0.2, n)
    segments.append(pl.DataFrame({"segment": name, "price": p_seg, "quantity": q_seg}))
bookings = pl.concat(segments)

per_segment = pricing.elasticity.segment_elasticity(
    bookings, price="price", quantity="quantity", segment="segment"
)
per_segment

# %% [markdown]
# ## 4. Dynamic elasticity — is the number even stable?
# Markets drift: competitors enter, mix shifts, customers learn. We simulate two years where
# sensitivity moves from -1.4 to -2.4 halfway, then watch the rolling fit track it and let the
# drift test make the call. A drifted elasticity silently invalidates last quarter's "optimal"
# price — this is the re-optimization trigger.

# %%
weeks = 208
drifting_e = np.where(np.arange(weeks) < weeks // 2, -1.4, -2.4)
p_t = rng.uniform(60.0, 140.0, weeks)
q_t = np.exp(8.0) * p_t**drifting_e * rng.lognormal(0.0, 0.15, weeks)
history = pl.DataFrame({"week": np.arange(weeks), "price": p_t, "quantity": q_t})

rolling = pricing.elasticity.rolling_elasticity(
    history, price="price", quantity="quantity", time="week", window=60
)
interactive.line(
    rolling.unpivot(on=["elasticity", "ci_low", "ci_high"], index="week"),
    "week",
    "value",
    color="variable",
    title="Rolling 60-week elasticity — the regime change is visible",
)

# %%
drift = pricing.elasticity.elasticity_drift(
    history, price="price", quantity="quantity", time="week"
)
print(
    f"baseline {drift.baseline.elasticity:.2f} -> recent {drift.recent.elasticity:.2f}  "
    f"(z={drift.z:.1f}, p={drift.p_value:.2g}) drifted={drift.drifted}"
)
print("=> re-run the price optimization on the recent window only")

# %% [markdown]
# ## 5. Nonlinear elasticity — one number can be wrong everywhere
# Constant elasticity says a 1% price move always shifts demand by e%. Real curves often steepen
# at high prices. The quadratic-in-logs check tests exactly that; when it fires, read the *local*
# elasticity at the prices you actually charge instead of one global slope.

# %%
log_p = np.log(p_t)
q_curved = np.exp(8.0 - 0.8 * log_p - 0.35 * log_p**2) * rng.lognormal(0.0, 0.12, weeks)
check = pricing.elasticity.nonlinear_elasticity_check(p_t, q_curved)
print(
    f"curvature {check.curvature:.2f} (p={check.p_value:.2g}), "
    f"AIC linear {check.aic_linear:.0f} vs quadratic {check.aic_quadratic:.0f} "
    f"-> nonlinear={check.nonlinear}"
)
for level in (70.0, 100.0, 130.0):
    print(f"  local elasticity at €{level:.0f}: {check.local_elasticity([level])[0]:.2f}")

# %% [markdown]
# ## 6. Why did *portfolio* elasticity move? Within vs mix
# Aggregate elasticity is a revenue-weighted mean over segments, so it moves for two very
# different reasons: segments became more price-sensitive (**within** — a pricing problem), or
# revenue shifted toward sensitive segments (**mix** — a portfolio problem). The shift-share
# decomposition splits the change exactly.

# %%
before = per_segment.select("segment", "elasticity").with_columns(
    pl.Series("weight", [400_000.0, 600_000.0])  # revenue weights last year
)
after = before.with_columns(
    pl.Series("elasticity", [-1.0, -2.7]),  # leisure got slightly more sensitive
    pl.Series("weight", [250_000.0, 750_000.0]),  # and grew its revenue share
)
decomposition = pricing.elasticity.elasticity_decomposition(before, after)
decomposition

# %%
total = decomposition["total"].sum()
within = decomposition["within"].sum()
mix = decomposition["mix"].sum()
print(f"aggregate elasticity change {total:+.3f} = within {within:+.3f} + mix {mix:+.3f}")
print("mostly mix -> the answer is portfolio/acquisition strategy, not blanket price cuts")

# %% [markdown]
# ## 7. Willingness-to-pay from buy/no-buy decisions
# Every quote shown is an experiment: offer a price, observe purchase. The logit purchase model
# turns those binary outcomes into a full WTP distribution (logistic, closed-form quantiles) —
# the median is the mass-market price, the upper quantiles are the premium tier's room.

# %%
offers = rng.uniform(50.0, 160.0, 6000)
wtp_true = rng.logistic(105.0, 14.0, 6000)  # median €105, scale 14
purchased = (wtp_true >= offers).astype(float)

logit = pricing.demand.fit_logit_demand(offers, purchased)
print(f"median WTP €{logit.wtp_median:.0f} (true 105)")
pricing.demand.willingness_to_pay(offers, purchased)

# %%
price_grid = np.linspace(50.0, 160.0, 111)
schedule = pricing.demand.demand_schedule(
    lambda p: logit.predict(p) * 10_000,
    price_grid,  # 10k prospects in the market
)
interactive.line(
    schedule.unpivot(on=["quantity", "revenue"], index="price"),
    "price",
    "value",
    color="variable",
    title="Purchase-probability demand: expected buyers and revenue by price",
)

# %% [markdown]
# ## 8. Van Westendorp — pricing *before* transaction data exists
# For a new product there are no purchase logs; the price sensitivity meter asks each respondent
# four prices (too cheap / cheap / expensive / too expensive) and reads the acceptable range off
# the curve crossings. Stated preference, so calibrate against the transaction-based WTP above
# once sales start.

# %%
anchor = rng.normal(105.0, 18.0, 600)
survey = pricing.demand.van_westendorp(
    too_cheap=anchor - 35.0 + rng.normal(0, 4, 600),
    cheap=anchor - 18.0 + rng.normal(0, 4, 600),
    expensive=anchor + 16.0 + rng.normal(0, 4, 600),
    too_expensive=anchor + 33.0 + rng.normal(0, 4, 600),
)
low, high = survey.acceptable_range
print(f"acceptable range €{low:.0f} … €{high:.0f}")
print(f"optimal price point €{survey.optimal_price:.0f}  (survey)")
print(f"transaction-based median WTP €{logit.wtp_median:.0f}  -> the two roughly agree")

# %%
interactive.line(
    survey.curves.unpivot(on=["too_cheap", "cheap", "expensive", "too_expensive"], index="price"),
    "price",
    "value",
    color="variable",
    title="Van Westendorp price sensitivity meter",
)

# %% [markdown]
# **Takeaways:** the headline elasticity (-1.8) is identified with a tight CI that excludes -1,
# so the directional pricing call is safe; demand is a substitute to the rival's fare (+0.9) and
# complementary to the add-on (-0.5), so the two prices must move *together*; business (-0.9) and
# leisure (-2.6) deserve different price logic entirely; the rolling fit + drift test caught the
# mid-history regime change (re-optimize on the recent window); curvature means quoting one
# elasticity across the whole price range would mislead at the edges; the portfolio elasticity
# moved mostly through *mix*, which is an acquisition-strategy answer, not a discount; and two
# independent WTP reads — purchase logits (€105 median) and the Van Westendorp survey — agree on
# the same acceptable corridor, which is exactly the cross-validation you want before committing
# a price list.
