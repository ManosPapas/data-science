"""Tests for modeling.monitor (drift detection)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.modeling import monitor


def test_psi_low_for_same_distribution(rng: np.random.Generator) -> None:
    baseline = rng.normal(0.0, 1.0, 4000)
    fresh = rng.normal(0.0, 1.0, 4000)
    assert monitor.psi(baseline, fresh) < 0.1


def test_psi_high_for_shifted_distribution(rng: np.random.Generator) -> None:
    baseline = rng.normal(0.0, 1.0, 4000)
    shifted = rng.normal(1.0, 1.0, 4000)
    assert monitor.psi(baseline, shifted) > 0.2


def test_psi_rejects_constant_baseline() -> None:
    with pytest.raises(ValueError, match="variation"):
        monitor.psi(np.full(100, 7.0), np.full(100, 7.0))


def test_ks_drift(rng: np.random.Generator) -> None:
    baseline = rng.normal(0.0, 1.0, 800)
    assert monitor.ks_drift(baseline, rng.normal(0.0, 1.0, 800)).p_value > 0.01
    assert monitor.ks_drift(baseline, rng.normal(2.0, 1.0, 800)).p_value < 0.001


def test_label_drift_calm_then_fires(rng: np.random.Generator) -> None:
    baseline = rng.choice(["a", "b"], size=2000, p=[0.7, 0.3])
    same_mix = rng.choice(["a", "b"], size=2000, p=[0.7, 0.3])
    shifted_mix = rng.choice(["a", "b"], size=2000, p=[0.5, 0.5])
    assert monitor.label_drift(baseline, same_mix).p_value > 0.01
    assert monitor.label_drift(baseline, shifted_mix).p_value < 0.001


def test_label_drift_flags_a_brand_new_class() -> None:
    baseline = np.array(["a", "b"] * 500)
    fresh = np.array(["a", "b", "c"] * 300)
    assert monitor.label_drift(baseline, fresh).p_value < 0.001


def test_drift_report_flags_the_drifted_column(rng: np.random.Generator) -> None:
    baseline = pl.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 2, 1000)})
    current = pl.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(9, 2, 1000)})
    report = monitor.drift_report(baseline, current)
    assert report.height == 2
    assert report["column"][0] == "b"  # sorted by psi, the shifted column first
    assert bool(report.filter(pl.col("column") == "b")["drifted"][0])
    assert not bool(report.filter(pl.col("column") == "a")["drifted"][0])
