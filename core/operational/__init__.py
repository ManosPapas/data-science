"""Operational ML / monitoring — feed readiness, sequential risk scoring, alerts, alert ROI.

The layer between a fitted model and a live process: is the data there to score
(:func:`feed_readiness`), update the risk as state arrives (:func:`rescore_sequence`), turn risk
+ remaining time into an action (:func:`generate_alerts`), and prove it paid off
(:func:`alert_metrics` / :func:`intervention_roi`). Generic over any checkpointed operational
process — generalized from an airline baggage digital twin.
"""

from core.operational.alerts import generate_alerts
from core.operational.evaluation import AlertMetrics, alert_metrics, intervention_roi
from core.operational.readiness import entities_missing, feed_readiness
from core.operational.risk_scoring import rescore_sequence

__all__ = [
    "AlertMetrics",
    "alert_metrics",
    "entities_missing",
    "feed_readiness",
    "generate_alerts",
    "intervention_roi",
    "rescore_sequence",
]
