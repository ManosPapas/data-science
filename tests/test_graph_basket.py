"""Tests for analytics.graph (network analytics) and analytics.basket (association rules)."""

from __future__ import annotations

import polars as pl
import pytest

from core.analytics import basket, graph


def _chain() -> pl.DataFrame:
    return pl.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})


def test_degree_centrality_finds_hub() -> None:
    table = graph.degree_centrality(_chain())
    assert table["node"][0] == "b"
    assert table.filter(pl.col("node") == "b")["degree"][0] == 2.0
    directed = graph.degree_centrality(_chain(), directed=True)
    assert {"in_degree", "out_degree"} <= set(directed.columns)


def test_pagerank_sums_to_one_and_ranks_sink() -> None:
    edges = pl.DataFrame({"source": ["a", "b", "c"], "target": ["c", "c", "a"]})
    ranks = graph.pagerank(edges)
    assert ranks["pagerank"].sum() == pytest.approx(1.0)
    assert ranks["node"][0] == "c"  # everyone points at c


def test_connected_components_split_islands() -> None:
    edges = pl.DataFrame({"source": ["a", "c"], "target": ["b", "d"]})
    table = graph.connected_components(edges)
    assert table["component"].n_unique() == 2
    assert table["component_size"].to_list() == [2, 2, 2, 2]


def test_shortest_paths_uses_weights() -> None:
    edges = pl.DataFrame(
        {"source": ["a", "a", "b"], "target": ["b", "c", "c"], "cost": [1.0, 10.0, 2.0]}
    )
    table = graph.shortest_paths(edges, origin="a", weight="cost")
    distances = dict(zip(table["node"].to_list(), table["distance"].to_list(), strict=True))
    assert distances["c"] == pytest.approx(3.0)  # a->b->c beats a->c direct
    with pytest.raises(ValueError, match="not present"):
        graph.shortest_paths(edges, origin="zz")


def test_minimum_spanning_tree_picks_cheapest_edges() -> None:
    triangle = pl.DataFrame(
        {"source": ["a", "b", "a"], "target": ["b", "c", "c"], "w": [1.0, 2.0, 10.0]}
    )
    tree = graph.minimum_spanning_tree_edges(triangle, weight="w")
    assert tree.height == 2
    assert tree["weight"].sum() == pytest.approx(3.0)


def test_max_flow_bottleneck() -> None:
    edges = pl.DataFrame(
        {
            "source": ["s", "s", "a", "b"],
            "target": ["a", "b", "t", "t"],
            "cap": [10.0, 5.0, 4.0, 9.0],
        }
    )
    value, flows = graph.max_flow(edges, origin="s", sink="t", weight="cap")
    assert value == pytest.approx(9.0)  # min cut: 4 via a + 5 via b
    assert flows["flow"].sum() > 0


def _baskets() -> pl.DataFrame:
    rows = []
    for tx in range(100):
        rows.append({"order": tx, "item": "bread"})
        if tx % 2 == 0:
            rows.append({"order": tx, "item": "butter"})
        if tx % 10 == 0:
            rows.append({"order": tx, "item": "champagne"})
    return pl.DataFrame(rows)


def test_frequent_itemsets_support() -> None:
    sets = basket.frequent_itemsets(_baskets(), transaction="order", item="item", min_support=0.05)
    singles = sets.filter(pl.col("size") == 1)
    assert singles.filter(pl.col("items").list.contains("bread"))["support"][0] == 1.0
    pair = sets.filter(
        (pl.col("size") == 2)
        & pl.col("items").list.contains("bread")
        & pl.col("items").list.contains("butter")
    )
    assert pair["support"][0] == pytest.approx(0.5)


def test_association_rules_confidence_and_lift() -> None:
    rules = basket.association_rules(
        _baskets(), transaction="order", item="item", min_support=0.05, min_confidence=0.1
    )
    butter_to_bread = rules.filter(
        (pl.col("antecedent") == "butter") & (pl.col("consequent") == "bread")
    )
    assert butter_to_bread["confidence"][0] == pytest.approx(1.0)
    assert butter_to_bread["lift"][0] == pytest.approx(1.0)  # bread is in every basket anyway
    assert rules["confidence"].min() >= 0.1


def test_association_rules_raise_without_pairs() -> None:
    lonely = pl.DataFrame({"order": [1, 2], "item": ["a", "b"]})
    with pytest.raises(ValueError, match="no frequent pairs"):
        basket.association_rules(lonely, transaction="order", item="item")


# --- Review-fix regression tests ----------------------------------------------------------------


def test_frequent_itemsets_arbitrary_max_size() -> None:
    # max_size is a real knob (was hard-capped at 3): a 4-item bundle must surface
    rows = []
    for tx in range(50):
        for it in ["a", "b", "c", "d"]:  # all four in every basket
            rows.append({"order": tx, "item": it})
    df = pl.DataFrame(rows)
    sets = basket.frequent_itemsets(
        df, transaction="order", item="item", min_support=0.5, max_size=4
    )
    assert 4 in sets["size"].to_list()
    quad = sets.filter(pl.col("size") == 4)
    assert quad["support"][0] == pytest.approx(1.0)


def test_itemset_and_rule_supports_agree() -> None:
    # shared miner: pair support must match across both APIs
    df = _baskets()
    itemsets = basket.frequent_itemsets(df, transaction="order", item="item", min_support=0.05)
    rules = basket.association_rules(
        df, transaction="order", item="item", min_support=0.05, min_confidence=0.0
    )
    pair = itemsets.filter(
        (pl.col("size") == 2)
        & pl.col("items").list.contains("bread")
        & pl.col("items").list.contains("butter")
    )
    rule = rules.filter((pl.col("antecedent") == "butter") & (pl.col("consequent") == "bread"))
    assert rule["support"][0] == pytest.approx(pair["support"][0])


def test_undirected_weight_not_doubled_on_both_orientations() -> None:
    # an edge listed in both directions is one undirected edge; weight must not double
    edges = pl.DataFrame(
        {"source": ["a", "b", "a"], "target": ["b", "a", "c"], "w": [3.0, 3.0, 4.0]}
    )
    deg = graph.degree_centrality(edges, weight="w")
    a_weight = deg.filter(pl.col("node") == "a")["weighted_degree"][0]
    assert a_weight == pytest.approx(7.0)  # 3 (a-b, once) + 4 (a-c), not 10


def test_frequent_itemsets_keeps_boundary_exact_support() -> None:
    # support EXACTLY min_support must survive (float round-up must not drop it)
    rows = [{"order": t, "item": "x"} for t in range(7)]  # 7 of 100 transactions
    rows += [{"order": t, "item": "filler"} for t in range(7, 100)]
    df = pl.DataFrame(rows)
    sets = basket.frequent_itemsets(df, transaction="order", item="item", min_support=0.07)
    assert "x" in [items[0] for items in sets["items"].to_list()]
