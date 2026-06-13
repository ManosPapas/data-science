"""Tests for the operational package (readiness, risk scoring, alerts, evaluation) + cross-env."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from core import operational
from core.modeling import compare, train


def test_feed_readiness_coverage_and_missing_expected() -> None:
    log = pl.DataFrame(
        {
            "bag": ["b1", "b1", "b2", "b3"],
            "milestone": ["checkin", "load", "checkin", "checkin"],
        }
    )
    cov = operational.feed_readiness(
        log, entity="bag", milestone="milestone", expected=["checkin", "load", "arrival"]
    )
    by = dict(zip(cov["milestone"].to_list(), cov["coverage"].to_list(), strict=True))
    assert by["checkin"] == pytest.approx(1.0)  # all 3 bags
    assert by["load"] == pytest.approx(1 / 3)
    assert by["arrival"] == pytest.approx(0.0)  # expected but never seen


def test_feed_readiness_timestamp_recency() -> None:
    log = pl.DataFrame({"bag": ["b1", "b2"], "milestone": ["load", "load"], "ts": [10, 20]})
    cov = operational.feed_readiness(log, entity="bag", milestone="milestone", timestamp="ts")
    assert "latest" in cov.columns
    assert cov.filter(pl.col("milestone") == "load")["latest"][0] == 20


def test_entities_missing_required() -> None:
    log = pl.DataFrame({"bag": ["b1", "b1", "b2"], "milestone": ["checkin", "load", "checkin"]})
    missing = operational.entities_missing(
        log, entity="bag", milestone="milestone", required=["checkin", "load"]
    )
    assert missing["bag"].to_list() == ["b2"]
    assert missing["missing"][0].to_list() == ["load"]


def test_rescore_sequence_trajectory(rng: np.random.Generator) -> None:
    x = pl.DataFrame({"f": rng.normal(size=200), "id": range(200)})
    y = (x["f"].to_numpy() > 0).astype(int)
    model = train.fit(LogisticRegression(), x.select("f"), y)
    # two checkpoints: the second flips the feature sign for entity 0
    early = x.head(3)
    late = early.with_columns(
        pl.when(pl.col("id") == 0).then(5.0).otherwise(pl.col("f")).alias("f")
    )
    traj = operational.rescore_sequence(
        model, {"early": early, "late": late}, feature_columns=["f"], id_column="id"
    )
    assert set(traj["checkpoint"].unique()) == {"early", "late"}
    assert traj.height == 6
    assert {"checkpoint", "id", "risk"} <= set(traj.columns)


def test_rescore_sequence_validates_columns(rng: np.random.Generator) -> None:
    x = pl.DataFrame({"f": rng.normal(size=50)})
    y = (x["f"].to_numpy() > 0).astype(int)
    model = train.fit(LogisticRegression(), x, y)
    with pytest.raises(ValueError, match="missing feature columns"):
        operational.rescore_sequence(
            model, {"c": pl.DataFrame({"g": [1.0]})}, feature_columns=["f"]
        )


def test_generate_alerts_bands_and_lead_gate() -> None:
    df = pl.DataFrame({"risk": [0.9, 0.5, 0.1, 0.95], "lead": [10.0, 10.0, 10.0, 0.5]})
    out = operational.generate_alerts(
        df,
        score="risk",
        bands=[(0.8, "rush"), (0.4, "monitor")],
        lead_time="lead",
        min_lead=1.0,
    )
    actions = out["action"].to_list()
    assert actions[0] == "rush"  # high risk, enough lead
    assert actions[1] == "monitor"  # mid band
    assert actions[2] == "none"  # below all thresholds
    assert actions[3] == "too_late"  # high risk but lead < min_lead


def test_alert_metrics() -> None:
    alerted = [1, 1, 0, 0, 1]
    event = [1, 0, 1, 0, 1]
    lead = [5.0, 0.0, 0.0, 0.0, 3.0]
    m = operational.alert_metrics(alerted, event, lead_time=lead)
    assert m.detection_rate == pytest.approx(2 / 3)  # 2 of 3 events caught
    assert m.precision == pytest.approx(2 / 3)  # 2 of 3 alerts real
    assert m.mean_lead_time == pytest.approx(4.0)  # mean of [5, 3] on detected events


def test_intervention_roi() -> None:
    roi = operational.intervention_roi(
        events_detected=100.0,
        value_per_prevented=500.0,
        prevention_rate=0.8,
        interventions=300.0,
        cost_per_intervention=10.0,
    )
    assert roi["benefit"] == pytest.approx(40_000.0)  # 100 * 0.8 * 500
    assert roi["cost"] == pytest.approx(3_000.0)
    assert roi["net"] == pytest.approx(37_000.0)
    assert roi["roi"] == pytest.approx(37_000.0 / 3_000.0)


def test_cross_environment_matrix(rng: np.random.Generator) -> None:
    # two environments with OPPOSITE decision rules → off-diagonal transfer should be poor
    def env(flip: bool, n: int = 300) -> tuple[pl.DataFrame, np.ndarray]:
        x = pl.DataFrame({"f": rng.normal(size=n)})
        rule = x["f"].to_numpy() < 0 if flip else x["f"].to_numpy() > 0
        return x, rule.astype(int)

    environments = {"a": env(False), "b": env(True)}

    def accuracy(model: object, x: object, y: object) -> float:
        return float((model.predict(x) == y).mean())  # type: ignore[attr-defined]

    matrix = compare.cross_environment(lambda: LogisticRegression(), environments, scoring=accuracy)
    cells = {(r["train"], r["test"]): r["score"] for r in matrix.iter_rows(named=True)}
    assert cells[("a", "a")] > 0.9 and cells[("b", "b")] > 0.9  # in-domain strong
    assert cells[("a", "b")] < 0.3  # opposite rule → transfer fails, as it should


def test_generate_alerts_rejects_none_action_collision() -> None:
    df = pl.DataFrame({"risk": [0.9, 0.2]})
    with pytest.raises(ValueError, match="collides with none_action"):
        operational.generate_alerts(df, score="risk", bands=[(0.5, "none")])


def test_alert_metrics_rejects_misaligned_lead_time() -> None:
    with pytest.raises(ValueError, match="align"):
        operational.alert_metrics([1, 1, 0], [1, 0, 1], lead_time=[5.0, 3.0])


def test_rescore_sequence_tolerates_id_dtype_drift(rng: np.random.Generator) -> None:
    from sklearn.linear_model import LogisticRegression

    x = pl.DataFrame({"f": rng.normal(size=40)})
    y = (x["f"].to_numpy() > 0).astype(int)
    model = train.fit(LogisticRegression(), x, y)
    early = pl.DataFrame({"f": [0.1, -0.2], "id": pl.Series([1, 2], dtype=pl.Int32)})
    late = pl.DataFrame({"f": [0.3, -0.1], "id": pl.Series([1, 2], dtype=pl.Int64)})
    traj = operational.rescore_sequence(
        model, {"early": early, "late": late}, feature_columns=["f"], id_column="id"
    )
    assert traj.height == 4  # vertical_relaxed supertypes the id column instead of raising
