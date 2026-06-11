"""Tests for the KPI modules."""

from __future__ import annotations

from core.kpi import behaviour, financial, profit


def test_gross_margin() -> None:
    assert financial.gross_margin(100.0, 40.0) == 0.6


def test_conversion_rate() -> None:
    assert behaviour.conversion_rate(20.0, 100.0) == 0.2


def test_funnel_rows() -> None:
    out = behaviour.funnel([100, 50, 10], ["visit", "cart", "buy"])
    assert out.height == 3
    assert out["overall_conversion"].to_list()[-1] == 0.1


def test_expected_value() -> None:
    value = profit.expected_value([1, 0, 1], [1, 0, 0], costs={"tp": 10, "fn": -5})
    assert value == 5.0
