"""User & system behaviour KPIs — GA, marketing, email, product, system; plus a funnel table.

Aggregate counts first, then call these. ``nan`` on divide-by-zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


# --- Acquisition & engagement -------------------------------------------------------------------


def conversion_rate(conversions: float, sessions: float) -> float:
    """Conversions / sessions."""
    return _ratio(conversions, sessions)


def bounce_rate(bounces: float, sessions: float) -> float:
    """Single-page (bounced) sessions / sessions."""
    return _ratio(bounces, sessions)


def engagement_rate(engaged: float, sessions: float) -> float:
    """Engaged sessions / sessions."""
    return _ratio(engaged, sessions)


def pages_per_session(pageviews: float, sessions: float) -> float:
    """Average pages viewed per session."""
    return _ratio(pageviews, sessions)


def average_session_duration(total_seconds: float, sessions: float) -> float:
    """Average session length in seconds."""
    return _ratio(total_seconds, sessions)


def new_user_rate(new_users: float, total_users: float) -> float:
    """Share of users who are new."""
    return _ratio(new_users, total_users)


def returning_user_rate(returning_users: float, total_users: float) -> float:
    """Share of users who are returning."""
    return _ratio(returning_users, total_users)


def activation_rate(activated: float, signups: float) -> float:
    """Activated users / signups (reached the 'aha' action)."""
    return _ratio(activated, signups)


def stickiness(dau: float, mau: float) -> float:
    """DAU / MAU — how often monthly users show up daily."""
    return _ratio(dau, mau)


def retention_rate(returning: float, cohort_size: float) -> float:
    """Returning users / original cohort size."""
    return _ratio(returning, cohort_size)


def repeat_rate(repeat_customers: float, total_customers: float) -> float:
    """Customers with more than one purchase / total customers."""
    return _ratio(repeat_customers, total_customers)


def viral_coefficient(invites_per_user: float, conversion_per_invite: float) -> float:
    """K-factor = invites per user * conversions per invite (>1 means viral growth)."""
    return invites_per_user * conversion_per_invite


# --- Commerce funnel ----------------------------------------------------------------------------


def add_to_cart_rate(carts: float, sessions: float) -> float:
    """Carts created / sessions."""
    return _ratio(carts, sessions)


def checkout_rate(checkouts: float, carts: float) -> float:
    """Checkouts started / carts."""
    return _ratio(checkouts, carts)


def cart_abandonment_rate(purchases: float, carts: float) -> float:
    """Share of carts that didn't convert: (carts - purchases) / carts."""
    return _ratio(carts - purchases, carts)


# --- Paid media ---------------------------------------------------------------------------------


def click_through_rate(clicks: float, impressions: float) -> float:
    """Clicks / impressions."""
    return _ratio(clicks, impressions)


def cost_per_click(spend: float, clicks: float) -> float:
    """Average cost per click (CPC)."""
    return _ratio(spend, clicks)


def cost_per_acquisition(spend: float, conversions: float) -> float:
    """Average cost per conversion (CPA)."""
    return _ratio(spend, conversions)


def cost_per_mille(spend: float, impressions: float) -> float:
    """Cost per 1,000 impressions (CPM)."""
    return _ratio(spend, impressions) * 1000


# --- Email --------------------------------------------------------------------------------------


def open_rate(opens: float, delivered: float) -> float:
    """Opens / delivered."""
    return _ratio(opens, delivered)


def click_to_open_rate(clicks: float, opens: float) -> float:
    """Clicks / opens."""
    return _ratio(clicks, opens)


def unsubscribe_rate(unsubscribes: float, delivered: float) -> float:
    """Unsubscribes / delivered."""
    return _ratio(unsubscribes, delivered)


# --- Satisfaction & system ----------------------------------------------------------------------


def csat(satisfied: float, responses: float) -> float:
    """Customer satisfaction: satisfied responses / total responses."""
    return _ratio(satisfied, responses)


def nps(scores: ArrayLike) -> float:
    """Net Promoter Score from 0-10 ratings: %promoters (9-10) minus %detractors (0-6)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((np.mean(arr >= 9) - np.mean(arr <= 6)) * 100)


def task_success_rate(successes: float, attempts: float) -> float:
    """Successful task completions / attempts."""
    return _ratio(successes, attempts)


def error_rate(errors: float, requests: float) -> float:
    """Errors / requests."""
    return _ratio(errors, requests)


def uptime(up_seconds: float, total_seconds: float) -> float:
    """Availability: up time / total time."""
    return _ratio(up_seconds, total_seconds)


# --- Funnel -------------------------------------------------------------------------------------


def funnel(counts: Sequence[int], steps: Sequence[str]) -> pl.DataFrame:
    """Funnel table: per-step count, step-over-step conversion, and overall conversion."""
    values = list(counts)
    top = values[0] if values else 0
    rows = []
    for index, (step, count) in enumerate(zip(steps, values, strict=True)):
        previous = values[index - 1] if index else count
        rows.append(
            {
                "step": step,
                "count": count,
                "step_conversion": _ratio(count, previous),
                "overall_conversion": _ratio(count, top),
            }
        )
    return pl.DataFrame(rows)
