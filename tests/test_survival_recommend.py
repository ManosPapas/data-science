"""Tests for modeling.survival, modeling.recommend, and evaluate.ranking_metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.modeling import evaluate, recommend, survival


def _censored_exponential(
    rng: np.random.Generator, scale: float = 10.0, n: int = 3000, cutoff: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    true_lifetimes = rng.exponential(scale, n)
    observed = np.minimum(true_lifetimes, cutoff)
    events = (true_lifetimes <= cutoff).astype(float)
    return observed, events


def test_kaplan_meier_tracks_true_survival(rng: np.random.Generator) -> None:
    durations, events = _censored_exponential(rng)
    curve = survival.kaplan_meier(durations, events)
    assert curve["survival"].is_sorted(descending=True)
    at_scale = survival.survival_at(durations, events, [10.0])[0]
    assert abs(at_scale - np.exp(-1.0)) < 0.03  # S(scale) = e^-1 for exponential
    assert survival.survival_at(durations, events, [0.0])[0] == 1.0


def test_median_and_restricted_mean(rng: np.random.Generator) -> None:
    durations, events = _censored_exponential(rng)
    median = survival.median_survival(durations, events)
    assert abs(median - 10.0 * np.log(2.0)) < 0.7
    rmst = survival.restricted_mean_survival(durations, events, horizon=15.0)
    truth = 10.0 * (1.0 - np.exp(-1.5))  # integral of e^(-t/10) to 15
    assert abs(rmst - truth) < 0.4


def test_survival_validations() -> None:
    with pytest.raises(ValueError, match=r"0 .censored. or 1"):
        survival.kaplan_meier([1.0, 2.0], [0.0, 2.0])
    with pytest.raises(ValueError, match="no events"):
        survival.kaplan_meier([1.0, 2.0], [0.0, 0.0])


def test_cox_ph_finds_risk_driver(rng: np.random.Generator) -> None:
    n = 2000
    tickets = rng.poisson(2.0, n).astype(float)
    noise = rng.normal(0.0, 1.0, n)
    hazard_scale = 12.0 * np.exp(-0.35 * tickets)  # more tickets -> shorter life
    lifetime = rng.exponential(hazard_scale)
    observed = np.minimum(lifetime, 20.0)
    events = (lifetime <= 20.0).astype(float)
    df = pl.DataFrame({"tenure": observed, "churned": events, "tickets": tickets, "noise": noise})
    table = survival.cox_ph(df, duration="tenure", event="churned", x=["tickets", "noise"])
    tickets_row = table.filter(pl.col("term") == "tickets")
    assert tickets_row["hazard_ratio"][0] > 1.2
    assert tickets_row["p_value"][0] < 0.001
    noise_row = table.filter(pl.col("term") == "noise")
    assert noise_row["p_value"][0] > 0.01


def _interactions() -> pl.DataFrame:
    rows = [
        ("u1", "espresso"),
        ("u1", "grinder"),
        ("u2", "espresso"),
        ("u2", "grinder"),
        ("u3", "espresso"),
        ("u3", "grinder"),
        ("u4", "espresso"),
        ("u5", "tea"),
        ("u5", "kettle"),
    ]
    return pl.DataFrame({"user": [r[0] for r in rows], "item": [r[1] for r in rows]})


def test_recommender_suggests_co_purchased_item() -> None:
    model = recommend.ItemItemRecommender().fit(_interactions(), user="user", item="item")
    picks = model.recommend("u4")  # bought espresso only
    assert picks["item"][0] == "grinder"
    similar = model.similar_items("espresso")
    assert similar["item"][0] == "grinder"
    assert 0.0 < similar["similarity"][0] <= 1.0


def test_recommender_excludes_seen_and_validates() -> None:
    model = recommend.ItemItemRecommender().fit(_interactions(), user="user", item="item")
    picks = model.recommend("u1")  # owns both coffee items
    assert "espresso" not in picks["item"].to_list()
    with pytest.raises(ValueError, match="unknown user"):
        model.recommend("nobody")
    with pytest.raises(ValueError, match="call fit"):
        recommend.ItemItemRecommender().recommend("u1")


def test_popularity_baseline() -> None:
    top = recommend.popularity_baseline(_interactions(), item="item", k=2)
    assert top["item"][0] == "espresso"
    assert top.height == 2


def test_ranking_metrics_perfect_vs_reversed() -> None:
    relevance = np.array([[1.0, 1.0, 0.0, 0.0]])
    perfect = evaluate.ranking_metrics(relevance, np.array([[0.9, 0.8, 0.2, 0.1]]), k=2)
    assert perfect["ndcg@2"] == pytest.approx(1.0)
    assert perfect["precision@2"] == pytest.approx(1.0)
    assert perfect["recall@2"] == pytest.approx(1.0)
    assert perfect["mrr"] == pytest.approx(1.0)

    reversed_ = evaluate.ranking_metrics(relevance, np.array([[0.1, 0.2, 0.8, 0.9]]), k=2)
    assert reversed_["precision@2"] == 0.0
    assert reversed_["mrr"] == pytest.approx(1.0 / 3.0)


def test_ranking_metrics_skips_empty_queries() -> None:
    relevance = np.array([[1.0, 0.0], [0.0, 0.0]])
    scores = np.array([[0.9, 0.1], [0.5, 0.4]])
    result = evaluate.ranking_metrics(relevance, scores, k=1)
    assert result["queries"] == 1.0
    with pytest.raises(ValueError, match="no query"):
        evaluate.ranking_metrics(np.zeros((1, 3)), np.ones((1, 3)))
