"""User & system behaviour KPIs — GA, marketing, email, product, system; plus a funnel table.

Aggregate counts first, then call these. ``nan`` on divide-by-zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from core.kpi._util import safe_div

# --- Acquisition & engagement -------------------------------------------------------------------


def conversion_rate(conversions: float, sessions: float) -> float:
    """Conversions / sessions."""
    return safe_div(conversions, sessions)


def bounce_rate(bounces: float, sessions: float) -> float:
    """Single-page (bounced) sessions / sessions."""
    return safe_div(bounces, sessions)


def engagement_rate(engaged: float, sessions: float) -> float:
    """Engaged sessions / sessions."""
    return safe_div(engaged, sessions)


def pages_per_session(pageviews: float, sessions: float) -> float:
    """Average pages viewed per session."""
    return safe_div(pageviews, sessions)


def average_session_duration(total_seconds: float, sessions: float) -> float:
    """Average session length in seconds."""
    return safe_div(total_seconds, sessions)


def new_user_rate(new_users: float, total_users: float) -> float:
    """Share of users who are new."""
    return safe_div(new_users, total_users)


def returning_user_rate(returning_users: float, total_users: float) -> float:
    """Share of users who are returning."""
    return safe_div(returning_users, total_users)


def activation_rate(activated: float, signups: float) -> float:
    """Activated users / signups (reached the 'aha' action)."""
    return safe_div(activated, signups)


def stickiness(dau: float, mau: float) -> float:
    """DAU / MAU — how often monthly users show up daily."""
    return safe_div(dau, mau)


def retention_rate(returning: float, cohort_size: float) -> float:
    """Returning users / original cohort size."""
    return safe_div(returning, cohort_size)


def repeat_rate(repeat_customers: float, total_customers: float) -> float:
    """Customers with more than one purchase / total customers."""
    return safe_div(repeat_customers, total_customers)


def viral_coefficient(invites_per_user: float, conversion_per_invite: float) -> float:
    """K-factor = invites per user * conversions per invite (>1 means viral growth)."""
    return invites_per_user * conversion_per_invite


# --- Commerce funnel ----------------------------------------------------------------------------


def add_to_cart_rate(carts: float, sessions: float) -> float:
    """Carts created / sessions."""
    return safe_div(carts, sessions)


def checkout_rate(checkouts: float, carts: float) -> float:
    """Checkouts started / carts."""
    return safe_div(checkouts, carts)


def cart_abandonment_rate(purchases: float, carts: float) -> float:
    """Share of carts that didn't convert: (carts - purchases) / carts."""
    return safe_div(carts - purchases, carts)


# --- Paid media ---------------------------------------------------------------------------------


def click_through_rate(clicks: float, impressions: float) -> float:
    """Clicks / impressions."""
    return safe_div(clicks, impressions)


def cost_per_click(spend: float, clicks: float) -> float:
    """Average cost per click (CPC)."""
    return safe_div(spend, clicks)


def cost_per_acquisition(spend: float, conversions: float) -> float:
    """Average cost per conversion (CPA)."""
    return safe_div(spend, conversions)


def cost_per_mille(spend: float, impressions: float) -> float:
    """Cost per 1,000 impressions (CPM)."""
    return safe_div(spend, impressions) * 1000


# --- Email --------------------------------------------------------------------------------------


def open_rate(opens: float, delivered: float) -> float:
    """Opens / delivered."""
    return safe_div(opens, delivered)


def click_to_open_rate(clicks: float, opens: float) -> float:
    """Clicks / opens."""
    return safe_div(clicks, opens)


def unsubscribe_rate(unsubscribes: float, delivered: float) -> float:
    """Unsubscribes / delivered."""
    return safe_div(unsubscribes, delivered)


# --- Satisfaction & system ----------------------------------------------------------------------


def csat(satisfied: float, responses: float) -> float:
    """Customer satisfaction: satisfied responses / total responses."""
    return safe_div(satisfied, responses)


def nps(scores: ArrayLike) -> float:
    """Net Promoter Score from 0-10 ratings: %promoters (9-10) minus %detractors (0-6)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((np.mean(arr >= 9) - np.mean(arr <= 6)) * 100)


def task_success_rate(successes: float, attempts: float) -> float:
    """Successful task completions / attempts."""
    return safe_div(successes, attempts)


def error_rate(errors: float, requests: float) -> float:
    """Errors / requests."""
    return safe_div(errors, requests)


def uptime(up_seconds: float, total_seconds: float) -> float:
    """Availability: up time / total time."""
    return safe_div(up_seconds, total_seconds)


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
                "step_conversion": safe_div(count, previous),
                "overall_conversion": safe_div(count, top),
            }
        )
    return pl.DataFrame(rows)
