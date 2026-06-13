"""Alert decision layer — turn a risk score + remaining lead time into an action.

Operational systems ask one question over and over: *what do we do now, and how much time is
left to do it?* This maps ``risk(t) → time_to_event(t) → action(t)``: a band ladder picks the
action by severity, and anything past the point of no return (too little lead time left) is
downgraded. Generalized from baggage rush/express/monitor alerting — the same shape fits fraud
holds, supply-chain expedites, QA interventions, churn saves.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def generate_alerts(
    df: pl.DataFrame,
    *,
    score: str,
    bands: Sequence[tuple[float, str]],
    lead_time: str | None = None,
    min_lead: float = 0.0,
    expired_action: str = "too_late",
    none_action: str = "none",
) -> pl.DataFrame:
    """Assign an action per row from a risk-score band ladder, gated by remaining lead time.

    ``bands`` is ``[(threshold, action), ...]``; each row takes the action of the highest
    threshold its ``score`` clears (order-independent — sorted internally). Rows below every
    threshold get ``none_action``. If ``lead_time`` is given, any actionable row with
    ``lead_time < min_lead`` is past the point of no return and becomes ``expired_action`` — the
    distinction between "act" and "too late to act" that makes an alert operational. Returns the
    frame with an ``action`` column appended.
    """
    if not bands:
        raise ValueError("need at least one (threshold, action) band")
    if any(label == none_action for _, label in bands):
        raise ValueError(
            f"a band label collides with none_action={none_action!r} — it would be exempt from the "
            "lead-time expiry gate; rename one of them"
        )
    ordered = sorted(bands, key=lambda b: b[0])  # ascending; later overrides assign the top band
    action = pl.lit(none_action)
    for threshold, label in ordered:
        action = pl.when(pl.col(score) >= threshold).then(pl.lit(label)).otherwise(action)
    result = df.with_columns(action.alias("action"))
    if lead_time is not None:
        result = result.with_columns(
            pl.when((pl.col("action") != none_action) & (pl.col(lead_time) < min_lead))
            .then(pl.lit(expired_action))
            .otherwise(pl.col("action"))
            .alias("action")
        )
    return result
