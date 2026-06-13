"""Alert backtesting & intervention ROI — did the alerts catch the events, and was it worth it?

The post-deployment proof every operational ML system owes its sponsor: replay alerts against
what actually happened (detection, precision, lead time), then translate to money under an
explicit cost model. The metrics are general; the cost numbers are yours to supply — keep them
visible, not buried. Generalized from baggage incident-replay + ROI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class AlertMetrics:
    """How well an alerting system caught the events it was meant to catch."""

    detection_rate: float  # share of true events that fired an alert (recall)
    precision: float  # share of alerts that were real events
    n_events: int
    n_alerts: int
    mean_lead_time: float  # average warning time on detected events (NaN if none / no lead data)


def alert_metrics(
    alerted: ArrayLike, event: ArrayLike, *, lead_time: ArrayLike | None = None
) -> AlertMetrics:
    """Detection rate, precision, and mean lead time from per-entity alert/event flags.

    ``alerted`` and ``event`` are 0/1 per entity. Detection rate (recall) is the operational
    headline — a missed event is the expensive failure; precision is the false-alarm cost.
    ``lead_time`` (warning time per entity) is averaged over *detected* events: an alert that
    fires too late to act on is not really a catch, so read lead time beside detection.
    """
    a = np.asarray(alerted, dtype=float).astype(bool)
    e = np.asarray(event, dtype=float).astype(bool)
    if a.shape != e.shape:
        raise ValueError("alerted and event must have the same length")
    n_events = int(e.sum())
    n_alerts = int(a.sum())
    detected = a & e
    detection_rate = float(detected.sum() / n_events) if n_events else float("nan")
    precision = float(detected.sum() / n_alerts) if n_alerts else float("nan")
    if lead_time is not None and detected.any():
        mean_lead = float(np.asarray(lead_time, dtype=float)[detected].mean())
    else:
        mean_lead = float("nan")
    return AlertMetrics(detection_rate, precision, n_events, n_alerts, mean_lead)


def intervention_roi(
    *,
    events_detected: float,
    value_per_prevented: float,
    prevention_rate: float = 1.0,
    interventions: float,
    cost_per_intervention: float,
    fixed_cost: float = 0.0,
) -> dict[str, float]:
    """Net value of acting on alerts: prevented losses minus the cost of intervening.

    ``events_detected`` * ``prevention_rate`` * ``value_per_prevented`` is the benefit (not every
    intervention on a detected event succeeds — ``prevention_rate`` is the realistic discount);
    ``interventions`` * ``cost_per_intervention`` + ``fixed_cost`` is the spend. Returns benefit,
    cost, net, and ROI. Every input is a number you must justify — that is the point of keeping
    the cost model explicit rather than hard-coding placeholder euros.
    """
    benefit = events_detected * prevention_rate * value_per_prevented
    cost = interventions * cost_per_intervention + fixed_cost
    net = benefit - cost
    return {
        "benefit": float(benefit),
        "cost": float(cost),
        "net": float(net),
        "roi": float(net / cost) if cost else float("nan"),
    }
