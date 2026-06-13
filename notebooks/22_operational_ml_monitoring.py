# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 22 · Operational ML — from a fitted model to a live, monitored, paying intervention
#
# Notebooks 03/21 build and govern a model; this one runs it against a *process*. The operational
# layer answers four questions a deployment actually faces: do we have the data to score
# (`feed_readiness`), how does risk evolve as an entity moves through checkpoints
# (`rescore_sequence`), what do we *do* and is there still time (`generate_alerts`), and did acting
# pay off (`alert_metrics` + `intervention_roi`). It closes with the generalization check every
# engagement needs — `compare.cross_environment`: does the model hold up outside the regime it was
# trained on? Synthetic shipment-tracking data with a known risk process, so every number is
# checkable. The pattern generalizes to any checkpointed process: deliveries, claims, loans,
# clinical pathways.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. A checkpointed process with a live event feed
# 4,000 shipments, each emitting milestone events (booked → collected → in_transit →
# out_for_delivery → delivered). Scans drop out — the feed is imperfect, as in production. The
# outcome we predict at booking: will the shipment be *late*? Risk rises with declared distance and
# a congested origin hub.

# %%
n = 4000
distance = rng.uniform(50.0, 2000.0, n)
congested_hub = (rng.random(n) < 0.3).astype(float)
logit = -2.0 + 0.0015 * distance + 1.1 * congested_hub
late = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
shipments = pl.DataFrame(
    {"shipment": [f"s{i}" for i in range(n)], "distance": distance, "congested_hub": congested_hub}
)

# build the event log: every shipment is booked; scans drop out with milestone-specific rates
milestones = ["booked", "collected", "in_transit", "out_for_delivery", "delivered"]
drop_rate = {
    "booked": 0.0,
    "collected": 0.05,
    "in_transit": 0.15,
    "out_for_delivery": 0.25,
    "delivered": 0.30,
}
events = []
for step, milestone in enumerate(milestones):
    keep = rng.random(n) >= drop_rate[milestone]
    for i in np.flatnonzero(keep):
        events.append({"shipment": f"s{i}", "milestone": milestone, "ts": step})
event_log = pl.DataFrame(events)
print(f"{event_log.height:,} milestone events across {n:,} shipments")

# %% [markdown]
# ## 2. Feed readiness — can we even score live?
# Before trusting a real-time model, check coverage. `out_for_delivery` and `delivered` scans
# arrive for only ~70-75% of shipments — a model that *needs* them can't score those entities yet.
# `entities_missing` lists exactly which shipments are short a required scan.

# %%
operational.feed_readiness(
    event_log, entity="shipment", milestone="milestone", expected=milestones, timestamp="ts"
)

# %%
not_ready = operational.entities_missing(
    event_log, entity="shipment", milestone="milestone", required=["booked", "collected"]
)
print(
    f"{not_ready.height:,} shipments missing a required early scan — hold or chase before scoring"
)

# %% [markdown]
# ## 3. Train the booking-time model, then re-score as the journey unfolds
# We fit on booking-time features (no leakage), then re-score the same shipments at successive
# checkpoints as congestion information firms up. The risk *trajectory* is what operations watch —
# a shipment trending up toward its deadline is the one to expedite.

# %%
labelled = shipments.select("distance", "congested_hub").with_columns(pl.Series("late", late))
train_full, test_full = split.train_test_split(labelled, test_size=0.3, stratify="late", seed=42)
train_df, test_df = train_full.drop("late"), test_full.drop("late")
y_train, y_test = train_full["late"].to_numpy(), test_full["late"].to_numpy()
model = train.fit(
    registry.make_model("gradient_boosting", task="classification"), train_df, y_train
)

# three checkpoints for a sample of shipments: congestion signal sharpens as the bag moves
sample = test_df.head(6).with_columns(pl.Series("shipment", [f"t{i}" for i in range(6)]))
snapshots = {
    "booked": sample,
    "in_transit": sample.with_columns((pl.col("congested_hub") * 1.0).alias("congested_hub")),
    "out_for_delivery": sample.with_columns(
        pl.when(pl.col("distance") > 1000)
        .then(1.0)
        .otherwise(pl.col("congested_hub"))
        .alias("congested_hub")
    ),
}
trajectory = operational.rescore_sequence(
    model, snapshots, feature_columns=["distance", "congested_hub"], id_column="shipment"
)
trajectory.pivot(on="checkpoint", index="shipment", values="risk")

# %% [markdown]
# ## 4. Alerts — what to do now, and is there still time
# Risk alone isn't an action. The band ladder maps risk → expedite / watch / none, and the
# lead-time gate downgrades anything past the point of no return: a 0.9-risk shipment with 2 hours
# left is no longer "expedite", it's "too_late" (escalate / pre-empt the claim instead).

# %%
scored = test_df.with_columns(
    pl.Series("risk", train.predict_proba(model, test_df)[:, 1]),
    pl.Series("hours_to_deadline", rng.uniform(0.0, 48.0, test_df.height)),
)
alerts = operational.generate_alerts(
    scored,
    score="risk",
    bands=[(0.6, "expedite"), (0.3, "watch")],
    lead_time="hours_to_deadline",
    min_lead=6.0,
)
print(alerts.group_by("action").len().sort("len", descending=True))

# %% [markdown]
# ## 5. Did it pay? Backtest the alerts, then price the intervention
# Replay the alerts against what actually happened: detection rate (the operational headline — a
# missed late shipment is the costly miss), precision (false-alarm cost), and lead time. Then the
# ROI under an explicit, defensible cost model — never a buried placeholder.

# %%
alerted = (alerts["action"].is_in(["expedite", "watch"])).cast(pl.Int8).to_numpy()
metrics = operational.alert_metrics(
    alerted, y_test, lead_time=scored["hours_to_deadline"].to_numpy()
)
print(
    f"detection {metrics.detection_rate:.0%}, precision {metrics.precision:.0%}, "
    f"mean lead {metrics.mean_lead_time:.1f}h on {metrics.n_events} late shipments"
)

# %%
detected = int(metrics.detection_rate * metrics.n_events)
roi = operational.intervention_roi(
    events_detected=detected,
    value_per_prevented=40.0,  # cost of a late-delivery claim avoided
    prevention_rate=0.7,  # expediting rescues 70% of flagged-and-caught shipments
    interventions=int(alerted.sum()),
    cost_per_intervention=3.0,  # expedite handling cost
)
print(
    f"benefit €{roi['benefit']:,.0f}, cost €{roi['cost']:,.0f}, "
    f"net €{roi['net']:,.0f}, ROI {roi['roi']:.0%}"
)

# %% [markdown]
# ## 6. Does the model generalize? Cross-environment matrix
# The model was trained on all hubs pooled. Will it transfer to a *new* region with a different
# congestion profile? Train on each region, test on every other: the diagonal is in-domain, the
# off-diagonal is transfer. A model that aces home and collapses away is overfit to a regime — the
# row spread, not one AUC, is the deployment-readiness signal.

# %%
region = rng.integers(0, 2, n)  # two regions; region 1 has a steeper distance effect
late_region = (
    rng.uniform(0, 1, n)
    < 1.0 / (1.0 + np.exp(-(-2.0 + (0.0015 + 0.001 * region) * distance + 1.1 * congested_hub)))
).astype(int)
feats = shipments.select("distance", "congested_hub")
environments = {
    f"region_{r}": (feats.filter(region == r), late_region[region == r]) for r in (0, 1)
}


def auc(fitted: object, x: object, y: object) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, fitted.predict_proba(x)[:, 1]))  # type: ignore[attr-defined]


compare.cross_environment(
    lambda: registry.make_model("gradient_boosting", task="classification"),
    environments,
    scoring=auc,
)

# %% [markdown]
# **Takeaways:** the feed is only ~70% complete at the late milestones, so a delivery-time model
# can't score every shipment live — `feed_readiness` turns that from a silent failure into a
# go/no-go list; risk is a trajectory, not a number, and re-scoring at each checkpoint surfaces the
# shipments trending toward a breach in time to act; the alert layer separates "expedite" from
# "too late to expedite" using the lead-time gate, which is the distinction that makes monitoring
# operational; the backtest proves detection and the ROI model prices it under assumptions kept in
# plain sight; and the cross-environment matrix is the honesty check before rollout — strong on the
# diagonal, weaker off it means re-train or re-calibrate per region rather than ship one model
# everywhere. This is the bridge from "we have a model" to "it runs, it pays, and we know where it
# breaks."
