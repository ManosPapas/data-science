"""Tests for analytics.choice — conditional logit, conjoint, MaxDiff, and experimental designs."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.analytics import choice


def _simulate_mnl(
    rng: np.random.Generator, n_sets: int, n_alt: int, beta: list[float]
) -> pl.DataFrame:
    """Choice data from a known logit: random features, utilities + softmax sampling."""
    k = len(beta)
    coef = np.array(beta)
    rows: list[dict[str, float | int]] = []
    for set_id in range(n_sets):
        x = rng.normal(size=(n_alt, k))
        utility = x @ coef
        probs = np.exp(utility - utility.max())
        probs = probs / probs.sum()
        picked = int(rng.choice(n_alt, p=probs))
        for alt in range(n_alt):
            rows.append(
                {"set": set_id, "alt": alt, "chosen": int(alt == picked)}
                | {f"x{j}": float(x[alt, j]) for j in range(k)}
            )
    return pl.DataFrame(rows)


def test_conditional_logit_recovers_known_coefficients(rng: np.random.Generator) -> None:
    df = _simulate_mnl(rng, n_sets=4000, n_alt=3, beta=[1.5, -0.8])
    model = choice.fit_conditional_logit(
        df, choice="chosen", choice_set="set", features=["x0", "x1"]
    )
    coefs = dict(zip(model.coefficients["term"], model.coefficients["coef"], strict=True))
    assert abs(coefs["x0"] - 1.5) < 0.25
    assert abs(coefs["x1"] - (-0.8)) < 0.25
    assert 0.0 < model.mcfadden_r2 < 1.0
    assert model.n_choices == 4000
    # both effects are real and the signs come through
    assert (model.coefficients["p_value"] < 0.01).all()


def test_conditional_logit_rejects_bad_choice_structure() -> None:
    df = pl.DataFrame({"set": [0, 0, 0], "chosen": [1, 1, 0], "x": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="exactly one chosen"):
        choice.fit_conditional_logit(df, choice="chosen", choice_set="set", features=["x"])


def _simulate_cbc(rng: np.random.Generator, n_sets: int, n_alt: int) -> pl.DataFrame:
    """CBC choices from known part-worths and a price effect."""
    brands = {"A": 0.0, "B": 0.9, "C": -0.6}
    sizes = {"S": 0.0, "L": 0.7}
    price_levels = [10.0, 20.0, 30.0]
    price_coef = -0.04
    brand_names, size_names = list(brands), list(sizes)
    rows: list[dict[str, object]] = []
    for set_id in range(n_sets):
        alts = []
        for _ in range(n_alt):
            brand = brand_names[int(rng.integers(len(brand_names)))]
            size = size_names[int(rng.integers(len(size_names)))]
            price = price_levels[int(rng.integers(len(price_levels)))]
            utility = brands[brand] + sizes[size] + price_coef * price
            alts.append((brand, size, price, utility))
        utilities = np.array([a[3] for a in alts])
        probs = np.exp(utilities - utilities.max())
        probs = probs / probs.sum()
        picked = int(rng.choice(n_alt, p=probs))
        for alt, (brand, size, price, _) in enumerate(alts):
            rows.append(
                {
                    "set": set_id,
                    "chosen": int(alt == picked),
                    "brand": brand,
                    "size": size,
                    "price": price,
                }
            )
    return pl.DataFrame(rows)


def test_choice_based_conjoint_recovers_part_worths_importance_and_wtp(
    rng: np.random.Generator,
) -> None:
    df = _simulate_cbc(rng, n_sets=5000, n_alt=3)
    fit = choice.choice_based_conjoint(
        df,
        choice="chosen",
        choice_set="set",
        attributes=["brand", "size"],
        price="price",
        reference={"brand": "A", "size": "S"},
    )

    pw = {(r["attribute"], r["level"]): r["utility"] for r in fit.part_worths.iter_rows(named=True)}
    assert pw[("brand", "A")] == 0.0 and pw[("size", "S")] == 0.0  # references pinned to 0
    assert abs(pw[("brand", "B")] - 0.9) < 0.25
    assert abs(pw[("brand", "C")] - (-0.6)) < 0.25
    assert abs(pw[("size", "L")] - 0.7) < 0.25
    assert fit.price_coef is not None and abs(fit.price_coef - (-0.04)) < 0.02

    assert abs(float(fit.importance["importance"].sum()) - 1.0) < 1e-9
    assert fit.importance.row(0, named=True)["attribute"] == "brand"  # widest part-worth span

    # WTP for brand B ≈ 0.9 / 0.04 ≈ 22.5 currency units
    wtp = {
        (r["attribute"], r["level"]): r["wtp"]
        for r in fit.willingness_to_pay().iter_rows(named=True)
    }
    assert abs(wtp[("brand", "B")] - 0.9 / 0.04) < 8.0

    products = pl.DataFrame(
        {
            "name": ["premium", "value"],
            "brand": ["B", "C"],
            "size": ["L", "S"],
            "price": [30.0, 12.0],
        }
    )
    shares = fit.simulate(products, label="name")
    assert abs(float(shares["share"].sum()) - 1.0) < 1e-9
    assert set(shares["product"].to_list()) == {"premium", "value"}


def test_metric_conjoint_recovers_part_worths(rng: np.random.Generator) -> None:
    profiles = choice.full_factorial({"brand": ["A", "B", "C"], "size": ["S", "L"]})
    repeated = pl.concat([profiles] * 60)
    truth = {"A": 0.0, "B": 1.2, "C": -0.5, "S": 0.0, "L": 0.8}
    base = 5.0
    rating = [
        base + truth[r["brand"]] + truth[r["size"]] + float(rng.normal(0, 0.4))
        for r in repeated.iter_rows(named=True)
    ]
    df = repeated.with_columns(pl.Series("rating", rating))
    fit = choice.metric_conjoint(
        df, rating="rating", attributes=["brand", "size"], reference={"brand": "A", "size": "S"}
    )

    pw = {(r["attribute"], r["level"]): r["utility"] for r in fit.part_worths.iter_rows(named=True)}
    assert abs(pw[("brand", "B")] - 1.2) < 0.15
    assert abs(pw[("size", "L")] - 0.8) < 0.15
    assert fit.intercept is not None and abs(fit.intercept - base) < 0.2
    assert fit.r_squared is not None and fit.r_squared > 0.5
    assert fit.model is None  # metric conjoint is OLS, not a choice model


def test_maxdiff_counts_scores_a_hand_example() -> None:
    df = pl.DataFrame(
        {
            "set": [1, 1, 1, 2, 2, 2],
            "item": ["A", "B", "C", "B", "C", "D"],
            "best": [1, 0, 0, 1, 0, 0],
            "worst": [0, 0, 1, 0, 0, 1],
        }
    )
    scores = choice.maxdiff_counts(df, item_col="item", best_col="best", worst_col="worst")
    by_item = {r["item"]: r["score"] for r in scores.iter_rows(named=True)}
    assert by_item == {"A": 1.0, "B": 0.5, "C": -0.5, "D": -1.0}
    assert scores["item"].to_list()[0] == "A"  # sorted best-first


def test_maxdiff_logit_orders_items(rng: np.random.Generator) -> None:
    items = ["A", "B", "C", "D", "E"]
    true_utility = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
    design = choice.maxdiff_design(items, n_sets=400, items_per_set=4, seed=1)
    rows: list[dict[str, object]] = []
    for part in design.partition_by("set"):
        shown = part["item"].to_list()
        util = np.array([true_utility[i] for i in shown])
        best_p = np.exp(util - util.max())
        best = shown[int(rng.choice(len(shown), p=best_p / best_p.sum()))]
        worst_p = np.exp(-util - (-util).max())
        worst = shown[int(rng.choice(len(shown), p=worst_p / worst_p.sum()))]
        for item in shown:
            rows.append(
                {
                    "set": part["set"][0],
                    "item": item,
                    "best": int(item == best),
                    "worst": int(item == worst and item != best),
                }
            )
    df = pl.DataFrame(rows).filter(
        # keep only sets that still have exactly one best and one worst after the tie guard
        (pl.col("best").sum().over("set") == 1) & (pl.col("worst").sum().over("set") == 1)
    )
    scores = choice.maxdiff_logit(
        df, set_col="set", item_col="item", best_col="best", worst_col="worst"
    )
    ranking = scores.sort("utility", descending=True)["item"].to_list()
    assert ranking[0] == "A" and ranking[-1] == "E"
    assert abs(float(scores["share"].sum()) - 1.0) < 1e-9


def test_full_factorial_enumerates_every_combination() -> None:
    grid = choice.full_factorial({"brand": ["A", "B", "C"], "size": ["S", "L"], "tier": [1, 2, 3]})
    assert grid.shape == (18, 3)
    assert grid.unique().height == 18


def test_orthogonal_design_is_efficient_and_valid() -> None:
    levels = {"brand": ["A", "B", "C"], "size": ["S", "L"], "speed": ["lo", "hi"]}
    design = choice.orthogonal_design(levels, n_profiles=8, seed=3)
    assert design.height == 8
    assert set(design.columns) == {"brand", "size", "speed"}
    for attr, allowed in levels.items():
        assert set(design[attr].to_list()) == set(allowed)  # every level covered -> estimable
    eff = choice.d_efficiency(design, levels)
    assert eff > 0.0
    # an arbitrary slice that drops a brand level is rank-deficient; the D-optimal pick is not
    arbitrary = choice.full_factorial(levels)[:8]
    assert eff > choice.d_efficiency(arbitrary, levels)
    # more runs never make the optimum worse; the full 12-run set is the most efficient
    full_pick = choice.orthogonal_design(levels, n_profiles=12, seed=3)
    assert choice.d_efficiency(full_pick, levels) >= eff - 1e-9


def test_choice_design_builds_valid_tasks() -> None:
    profiles = choice.full_factorial({"brand": ["A", "B", "C"], "size": ["S", "L"]})
    design = choice.choice_design(profiles, n_sets=8, alternatives=3, seed=5)
    assert design.height == 8 * 3
    assert {"choice_set", "alternative", "brand", "size", "none"} <= set(design.columns)
    per_set = design.group_by("choice_set").agg(pl.len().alias("n"))
    assert (per_set["n"] == 3).all()

    with_none = choice.choice_design(profiles, n_sets=4, alternatives=2, include_none=True)
    assert with_none.filter(pl.col("none") == 1).height == 4


def test_maxdiff_design_is_balanced() -> None:
    design = choice.maxdiff_design(list("ABCDEFGHIJ"), n_sets=20, items_per_set=4, seed=7)
    assert design.height == 20 * 4
    per_set = design.group_by("set").agg(pl.col("item").n_unique().alias("u"))
    assert (per_set["u"] == 4).all()  # distinct items within a task
    exposure = design["item"].value_counts()["count"].to_numpy()
    assert exposure.max() - exposure.min() <= 2  # roughly even exposure


def test_maxdiff_logit_rejects_same_best_and_worst() -> None:
    df = pl.DataFrame(
        {
            "set": [1, 1, 1],
            "item": ["A", "B", "C"],
            "best": [1, 0, 0],
            "worst": [1, 0, 0],  # A flagged both best and worst — contradictory
        }
    )
    with pytest.raises(ValueError, match="both best and worst"):
        choice.maxdiff_logit(df, set_col="set", item_col="item", best_col="best", worst_col="worst")


def test_orthogonal_design_raises_cleanly_when_no_full_rank_design_found() -> None:
    # a 10-level attribute against the parameter-count floor can leave every restart singular;
    # the failure must surface as a clear ValueError, not a bare AssertionError
    with pytest.raises(ValueError, match="full-rank"):
        choice.orthogonal_design({"a": list(range(10)), "b": [0, 1]}, seed=14, restarts=10)


def test_simulate_rejects_an_unfitted_level(rng: np.random.Generator) -> None:
    df = _simulate_cbc(rng, n_sets=400, n_alt=3)
    fit = choice.choice_based_conjoint(
        df, choice="chosen", choice_set="set", attributes=["brand", "size"], price="price"
    )
    novel = pl.DataFrame({"brand": ["Z"], "size": ["S"], "price": [20.0]})  # brand Z never fitted
    with pytest.raises(ValueError, match="not in the fitted design"):
        fit.simulate(novel)
