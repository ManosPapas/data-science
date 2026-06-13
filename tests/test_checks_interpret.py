"""Tests for modeling.checks, modeling.interpret, validate.check_rules, monitor control charts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from core.features import validate
from core.modeling import checks, interpret, monitor, train


def _churn_model(rng: np.random.Generator) -> tuple[LogisticRegression, pl.DataFrame]:
    n = 1500
    x = pl.DataFrame(
        {
            "tickets": rng.poisson(2.0, n).astype(float),
            "spend": rng.uniform(10.0, 200.0, n),
        }
    )
    logit = -1.0 + 0.8 * x["tickets"].to_numpy() - 0.01 * x["spend"].to_numpy()
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    model = train.fit(LogisticRegression(max_iter=1000), x, y)
    return model, x


def test_monotonicity_pass_and_fail(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    increasing = checks.monotonicity(model, x, feature="tickets", direction="increasing")
    assert increasing.passed  # logistic with positive coef is monotone increasing
    wrong_way = checks.monotonicity(model, x, feature="tickets", direction="decreasing")
    assert wrong_way.violation_rate == 1.0
    assert wrong_way.worst_gap > 0


def test_expected_directions_table(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    table = checks.expected_directions(model, x, {"tickets": "increasing", "spend": "decreasing"})
    assert table["passed"].all()
    assert set(table.columns) == {"feature", "direction", "violation_rate", "worst_gap", "passed"}


def test_prediction_bounds(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    bounds = checks.prediction_bounds(model, x, lower=0.0, upper=1.0)
    assert bounds["share_below"] == 0.0
    assert bounds["share_above"] == 0.0
    assert 0.0 <= bounds["min"] <= bounds["max"] <= 1.0


def test_perturbation_stability(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    report = checks.perturbation_stability(model, x, scale=0.01, n_repeats=5)
    assert report.mean_abs_change < 0.05  # small noise should barely move a linear model
    assert report.p95_abs_change >= report.mean_abs_change
    assert 0.0 <= report.relative_mean_change < 1.0


def test_check_rules_counts_violations() -> None:
    df = pl.DataFrame({"price": [10.0, 5.0, None], "cost": [8.0, 6.0, 1.0]})
    rules = {
        "price covers cost": pl.col("price") >= pl.col("cost"),
        "price present": pl.col("price").is_not_null(),
    }
    table = validate.check_rules(df, rules)
    by_rule = dict(zip(table["rule"].to_list(), table["violations"].to_list(), strict=True))
    assert by_rule["price covers cost"] == 2  # the violation and the null both fail
    assert by_rule["price present"] == 1
    with pytest.raises(ValueError, match="business rules failed"):
        validate.check_rules(df, rules, raise_on_error=True)


def test_control_limits_and_ewma_alerts(rng: np.random.Generator) -> None:
    baseline = rng.normal(100.0, 5.0, 200)
    limits = monitor.control_limits(baseline)
    assert limits.lower < 100.0 < limits.upper

    stable = rng.normal(100.0, 5.0, 60)
    drifted = np.concatenate([stable, rng.normal(106.0, 5.0, 60)])  # ~1.2 sigma shift
    chart = monitor.ewma_alerts(drifted, baseline)
    alerts = chart["alert"].to_numpy()
    assert not alerts[:20].any()  # early stable stretch stays quiet
    assert alerts[80:].any()  # persistent small shift is caught
    assert {"value", "ewma", "lower", "upper", "alert"} <= set(chart.columns)


def test_counterfactual_finds_minimal_change(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    row = x[0].with_columns(pl.lit(8.0).alias("tickets"))  # a high-risk customer
    result = interpret.counterfactual(
        model,
        row,
        candidates={"tickets": np.arange(0.0, 9.0), "spend": np.array([50.0, 150.0, 200.0])},
        target=0.3,
        direction="<=",
    )
    assert result.found
    assert result.score <= 0.3
    assert result.baseline_score > 0.3
    assert "tickets" in result.changes  # tickets is the lever that matters


def test_counterfactual_no_op_when_target_met(rng: np.random.Generator) -> None:
    model, x = _churn_model(rng)
    row = x[0].with_columns(pl.lit(0.0).alias("tickets"), pl.lit(200.0).alias("spend"))
    result = interpret.counterfactual(
        model, row, candidates={"tickets": np.arange(0.0, 5.0)}, target=0.9, direction="<="
    )
    assert result.found
    assert result.changes == {}
    assert result.candidates_evaluated == 0


def test_conformal_intervals_cover(rng: np.random.Generator) -> None:
    n = 1200
    x = pl.DataFrame({"x1": rng.uniform(0.0, 10.0, n)})
    y = 3.0 * x["x1"].to_numpy() + rng.normal(0.0, 1.0, n)
    fitted = train.fit(LinearRegression(), x[:400], y[:400])
    intervals = interpret.conformal_intervals(fitted, x[400:800], y[400:800], x[800:], alpha=0.1)
    covered = (
        (intervals["lower"].to_numpy() <= y[800:]) & (y[800:] <= intervals["upper"].to_numpy())
    ).mean()
    assert covered >= 0.85  # guarantee is >= 0.9 in expectation; allow sampling slack
    assert (intervals["upper"] - intervals["lower"]).min() > 0


def test_conformal_needs_enough_calibration(rng: np.random.Generator) -> None:
    x = pl.DataFrame({"x1": rng.uniform(0.0, 1.0, 50)})
    y = x["x1"].to_numpy()
    fitted = train.fit(LinearRegression(), x, y)
    with pytest.raises(ValueError, match="calibration rows"):
        interpret.conformal_intervals(fitted, x[:5], y[:5], x, alpha=0.05)


def test_confidence_score_methods() -> None:
    probabilities = np.array([[0.95, 0.05], [0.55, 0.45]])
    margin = interpret.confidence_score(probabilities)
    assert margin[0] > margin[1]
    entropy = interpret.confidence_score(probabilities, method="entropy")
    assert entropy[0] > entropy[1]
    one_dim = interpret.confidence_score(np.array([0.95, 0.5]))
    assert one_dim[0] == pytest.approx(0.9)
    assert one_dim[1] == pytest.approx(0.0)
