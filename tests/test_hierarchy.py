"""Tests for forecasting.hierarchy (coherent forecast reconciliation)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.forecasting import hierarchy

HIERARCHY = {"total": ["EU", "NA"], "EU": ["UK", "DE"]}


def _base_forecasts() -> dict[str, np.ndarray]:
    return {
        "UK": np.array([10.0, 11.0]),
        "DE": np.array([20.0, 21.0]),
        "NA": np.array([35.0, 36.0]),
        "EU": np.array([33.0, 34.0]),  # disagrees with UK+DE = 30/32
        "total": np.array([70.0, 72.0]),  # disagrees with EU+NA
    }


def test_summing_matrix_shape_and_content() -> None:
    s, nodes, leaves = hierarchy.summing_matrix(HIERARCHY)
    assert leaves == ["NA", "UK", "DE"]
    assert s.shape == (5, 3)
    total_row = s[nodes.index("total")]
    assert total_row.tolist() == [1.0, 1.0, 1.0]
    eu_row = s[nodes.index("EU")]
    assert eu_row.tolist() == [0.0, 1.0, 1.0]


def test_coherence_error_reports_gaps() -> None:
    gaps = hierarchy.coherence_error(_base_forecasts(), HIERARCHY)
    assert gaps.filter(pl.col("node") == "EU")["mean_abs_gap"][0] == pytest.approx(2.5)


def _assert_coherent(result: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(result["EU"], result["UK"] + result["DE"], atol=1e-9)
    np.testing.assert_allclose(result["total"], result["EU"] + result["NA"], atol=1e-9)


def test_bottom_up_trusts_leaves() -> None:
    result = hierarchy.reconcile(_base_forecasts(), HIERARCHY, method="bottom_up")
    _assert_coherent(result)
    np.testing.assert_allclose(result["UK"], [10.0, 11.0])
    np.testing.assert_allclose(result["total"], [65.0, 68.0])


def test_top_down_splits_the_total() -> None:
    result = hierarchy.reconcile(
        _base_forecasts(),
        HIERARCHY,
        method="top_down",
        proportions={"UK": 0.2, "DE": 0.3, "NA": 0.5},
    )
    _assert_coherent(result)
    np.testing.assert_allclose(result["total"], [70.0, 72.0])
    np.testing.assert_allclose(result["UK"], [14.0, 14.4])


def test_ols_reconciliation_is_coherent_and_balanced() -> None:
    forecasts = _base_forecasts()
    result = hierarchy.reconcile(forecasts, HIERARCHY, method="ols")
    _assert_coherent(result)
    # projection: already-coherent forecasts pass through unchanged
    coherent_input = hierarchy.reconcile(result, HIERARCHY, method="ols")
    for node, values in result.items():
        np.testing.assert_allclose(coherent_input[node], values, atol=1e-9)
    # gaps after reconciliation are zero
    gaps = hierarchy.coherence_error(result, HIERARCHY)
    assert gaps["mean_abs_gap"].max() == pytest.approx(0.0, abs=1e-9)


def test_reconcile_validations() -> None:
    partial = {"UK": [1.0], "DE": [2.0]}
    with pytest.raises(ValueError, match="missing"):
        hierarchy.reconcile(partial, HIERARCHY, method="ols")
    with pytest.raises(ValueError, match="every leaf"):
        hierarchy.reconcile({"UK": [1.0]}, HIERARCHY, method="bottom_up")
    with pytest.raises(ValueError, match="sum to 1"):
        hierarchy.reconcile(
            _base_forecasts(),
            HIERARCHY,
            method="top_down",
            proportions={"UK": 0.5, "DE": 0.3, "NA": 0.5},
        )
