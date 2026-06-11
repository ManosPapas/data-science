"""Shared scalar helpers for the KPI modules."""

from __future__ import annotations


def safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning ``nan`` on a zero denominator so KPI helpers never raise."""
    return numerator / denominator if denominator else float("nan")
