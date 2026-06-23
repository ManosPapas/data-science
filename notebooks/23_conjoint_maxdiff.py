# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 23 · Conjoint & MaxDiff — what drives choice, what it's worth, and what would win
#
# You can't A/B test a fare bundle that doesn't exist yet. Conjoint shows respondents *hypothetical*
# products, watches what they pick, and recovers the value of each feature from the choices alone;
# MaxDiff does the same for a plain list of perks. This notebook runs the full loop end-to-end on
# an airline-fare study with **known truth**, so every estimate is checked against the answer:
# design the study → field it → estimate part-worths & importance → read willingness-to-pay →
# simulate market share for a candidate line-up → and separately rank a list of ancillaries by
# MaxDiff. The decisions it informs: which features to build, how to price them, and which bundle
# to launch against the competition.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Design the study (the *structures*)
# A fare has four attributes; testing every combination is a 48-profile full factorial. You don't
# need them all — `orthogonal_design` picks a **D-efficient** fraction that still identifies every
# part-worth, so the survey stays short. `d_efficiency` scores the balance (higher = tighter, less
# correlated estimates). The full factorial is the most balanced benchmark; the subset trades a
# little efficiency for a far shorter questionnaire.

# %%
attribute_levels = {
    "airline": ["Norse", "Legacy", "LCC"],
    "bag": ["none", "checked"],
    "seat": ["standard", "extra_legroom"],
    "fare": [39, 59, 79, 99],  # discrete in the design; enters the model as a continuous price
}
full = choice.full_factorial(attribute_levels)
efficient = choice.orthogonal_design(attribute_levels, n_profiles=16, seed=1)
print(
    f"full factorial: {full.height} profiles; field a {efficient.height}-profile D-efficient subset"
)
print(
    f"D-efficiency — full {choice.d_efficiency(full, attribute_levels):.3f}  "
    f"subset {choice.d_efficiency(efficient, attribute_levels):.3f}"
)

# %% [markdown]
# `choice_design` assembles profiles into the choice tasks a respondent sees (here 3 fares per
# screen). For clean truth-recovery below we draw tasks from the full profile space; in a live
# study you'd field the efficient subset above and `include_none=True` for a "wouldn't book any"
# outside option.

# %%
tasks = choice.choice_design(full, n_sets=2500, alternatives=3, seed=2)
tasks.head(6)

# %% [markdown]
# ## 2. Field it — simulate choices from known part-worths
# Truth we'll try to recover: a checked bag adds **+0.8** utility, extra legroom **+0.5**, Legacy is
# **+0.4** vs Norse and LCC **-0.7**; every £ of fare costs **-0.025**. Each respondent picks the
# highest-utility fare shown (utility + i.i.d. Gumbel noise = exactly the logit choice rule).

# %%
truth = {
    "Norse": 0.0,
    "Legacy": 0.4,
    "LCC": -0.7,
    "none": 0.0,
    "checked": 0.8,
    "standard": 0.0,
    "extra_legroom": 0.5,
}
price_coef_true = -0.025
util = (
    np.array([truth[a] for a in tasks["airline"].to_list()])
    + np.array([truth[b] for b in tasks["bag"].to_list()])
    + np.array([truth[s] for s in tasks["seat"].to_list()])
    + price_coef_true * tasks["fare"].to_numpy()
    + rng.gumbel(0.0, 1.0, tasks.height)
)
responses = tasks.with_columns(pl.Series("util", util)).with_columns(
    (pl.col("util") == pl.col("util").max().over("choice_set")).cast(pl.Int8).alias("chosen")
)
print(f"{responses['choice_set'].n_unique()} choice tasks · {responses.height} alternatives shown")

# %% [markdown]
# ## 3. Estimate — part-worths and attribute importance
# `choice_based_conjoint` dummy-codes the levels, fits the conditional logit, and hands back the
# part-worths (utility per level, reference pinned to 0), their importance, and a share simulator.
# The fitted price coefficient and part-worths land on the truth; the McFadden pseudo-R² runs
# well below an OLS R² (0.1-0.3 is typical for choice data) yet confirms clear signal.

# %%
cbc = choice.choice_based_conjoint(
    responses,
    choice="chosen",
    choice_set="choice_set",
    attributes=["airline", "bag", "seat"],
    price="fare",
    reference={"airline": "Norse", "bag": "none", "seat": "standard"},
)
print(
    f"McFadden pseudo-R² {cbc.model.mcfadden_r2:.3f} · "
    f"fitted price coef {cbc.price_coef:.4f} (true {price_coef_true})"
)
cbc.part_worths

# %%
cbc.importance

# %%
fig, axes = base.grid(2)
business.part_worth_utilities(
    cbc.part_worths, ax=axes[0], title="Part-worth utilities (Norse / none / standard = 0)"
)
business.attribute_importance(cbc.importance, ax=axes[1], title="Attribute importance")

# %% [markdown]
# ## 4. Willingness-to-pay — features in money
# Divide each part-worth by the (negative) price coefficient and the utilities become pounds: the
# fare a flyer would trade to get that feature. True values: checked bag £32, extra legroom £20,
# Legacy £16, LCC -£28 (i.e. you'd have to *discount* £28 to offset flying LCC vs Norse).

# %%
cbc.willingness_to_pay()

# %% [markdown]
# ## 5. Share of preference — what would actually win
# `simulate` applies the logit rule to a candidate line-up and returns each option's predicted
# booking share. This is the lever for product/pricing decisions: try a bundle against the
# competition, then run a what-if on price.

# %%
lineup = pl.DataFrame(
    {
        "product": ["Norse premium", "Legacy basic", "LCC barebones"],
        "airline": ["Norse", "Legacy", "LCC"],
        "bag": ["checked", "none", "none"],
        "seat": ["extra_legroom", "standard", "standard"],
        "fare": [79.0, 89.0, 39.0],
    }
)
shares = cbc.simulate(lineup, label="product")
shares

# %%
business.preference_share(shares, label="product", title="Share of preference — current line-up")

# %%
# What-if: drop the Norse fare £79 -> £59 and re-simulate.
cut = lineup.with_columns(
    pl.when(pl.col("product") == "Norse premium").then(59.0).otherwise(pl.col("fare")).alias("fare")
)
shares_cut = cbc.simulate(cut, label="product")
before = shares.filter(pl.col("product") == "Norse premium")["share"][0]
after = shares_cut.filter(pl.col("product") == "Norse premium")["share"][0]
print(f"cutting the Norse fare £79 -> £59 lifts our share {before:.1%} -> {after:.1%}")

# %% [markdown]
# ## 6. Ratings instead of choices — metric conjoint
# When respondents *rated* profiles rather than chose, the part-worths come straight out of an OLS:
# `metric_conjoint`. Lower-variance when ratings exist, and here it recovers the same structure as
# the choice-based fit — a useful cross-check.

# %%
rated = pl.concat([full] * 40)
rating = (
    5.0
    + np.array([truth[a] for a in rated["airline"].to_list()])
    + np.array([truth[b] for b in rated["bag"].to_list()])
    + np.array([truth[s] for s in rated["seat"].to_list()])
    + price_coef_true * rated["fare"].to_numpy()
    + rng.normal(0.0, 0.5, rated.height)
)
metric = choice.metric_conjoint(
    rated.with_columns(pl.Series("rating", rating)),
    rating="rating",
    attributes=["airline", "bag", "seat"],
    price="fare",
    reference={"airline": "Norse", "bag": "none", "seat": "standard"},
)
print(f"ratings R² {metric.r_squared:.3f} — part-worths agree with the choice-based fit:")
metric.part_worths

# %% [markdown]
# ## 7. MaxDiff — ranking a list of perks
# A different question: "of these 12 ancillaries, which matter most?" MaxDiff shows small subsets
# and asks for the **best** and **worst** in each. `maxdiff_design` builds balanced tasks;
# `maxdiff_counts` gives the quick best-minus-worst score and `maxdiff_logit` puts the items on an
# interval utility scale (reference item pinned to 0). Both recover the true ordering — lounge
# access on top, welcome drink at the bottom.

# %%
perk_value = {
    "lounge access": 2.1,
    "free checked bag": 1.6,
    "extra legroom": 1.2,
    "priority boarding": 0.7,
    "fast-track security": 0.4,
    "onboard wifi": 0.1,
    "free seat selection": -0.1,
    "meal voucher": -0.3,
    "miles bonus": -0.6,
    "flexible change": -0.9,
    "premium snack": -1.4,
    "welcome drink": -1.8,
}
perks = list(perk_value)
md_design = choice.maxdiff_design(perks, n_sets=600, items_per_set=4, seed=11)

md_rows: list[dict[str, object]] = []
for task in md_design.partition_by("set"):
    shown = task["item"].to_list()
    latent = np.array([perk_value[p] for p in shown]) + rng.gumbel(0.0, 1.0, len(shown))
    best, worst = shown[int(latent.argmax())], shown[int(latent.argmin())]
    for perk in shown:
        md_rows.append(
            {
                "set": task["set"][0],
                "item": perk,
                "best": int(perk == best),
                "worst": int(perk == worst),
            }
        )
md = pl.DataFrame(md_rows)

# %%
counts = choice.maxdiff_counts(md, item_col="item", best_col="best", worst_col="worst")
utilities = choice.maxdiff_logit(
    md, set_col="set", item_col="item", best_col="best", worst_col="worst"
)
print(
    f"counting winner: {counts['item'][0]} · logit winner: {utilities['item'][0]} "
    "(true top: lounge access)"
)
utilities

# %%
business.maxdiff_scores(
    utilities, item="item", value="utility", title="MaxDiff perk preferences (logit utilities)"
)

# %% [markdown]
# **Takeaways:** from choices alone the study recovers what each fare feature is worth — a checked
# bag (~£32) and extra legroom (~£20) are the levers worth charging for, while flying LCC instead of
# Norse needs a ~£28 discount to break even; price is the biggest single driver of choice. The share
# simulator turns those part-worths into a launch decision — our premium bundle's share and how a
# £20 fare cut moves it — without fielding a single real fare. Metric conjoint reproduces the same
# part-worths from ratings, and MaxDiff ranks the ancillary list (lounge access and a free bag on
# top) for the merchandising roadmap. All stated preference: calibrate against real bookings (see
# `pricing.demand` / `pricing.elasticity`) once the bundle is live.
