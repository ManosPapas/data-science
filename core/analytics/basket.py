"""Market basket analysis — frequent itemsets and association rules, in pure Polars.

What sells together: support (how often), confidence (P(B|A)), and lift (how much more often
than chance). Drives cross-sell placement, bundling, and the co-purchase graph
(``analytics.graph``).

Frequent itemsets are mined with Eclat: each itemset carries its *tidset* (the transactions that
contain it), and a (k+1)-set's tidset is the intersection of two k-sets' tidsets. That keeps the
work to one transaction scan plus list intersections — no repeated transaction self-joins — so it
scales to large baskets and arbitrary itemset sizes. Apriori pruning is implicit: only frequent
k-sets are extended.
"""

from __future__ import annotations

import polars as pl

_PREFIX_SEP = "\x1f"  # unit separator — safe to join item strings on


def _transactions(df: pl.DataFrame, transaction: str, item: str) -> tuple[pl.DataFrame, int]:
    base = df.select(transaction, item).drop_nulls().unique()
    n_tx = base[transaction].n_unique()
    if n_tx == 0:
        raise ValueError("no transactions found")
    return base, n_tx


def _frequent_levels(
    base: pl.DataFrame, *, transaction: str, item: str, n_tx: int, min_support: float, max_size: int
) -> list[pl.DataFrame]:
    """Eclat per-level frames (``items`` sorted List[str], ``count``, ``support``), k=1..max_size.

    Returns one frame per size with at least one frequent set; empties stop the ladder early.
    """
    # epsilon so a support of exactly min_support survives float round-up (0.07*100 = 7.0000...1)
    threshold = min_support * n_tx - 1e-9
    singles = (
        base.group_by(item)
        .agg(pl.col(transaction).unique().alias("_tids"))
        .with_columns(pl.col("_tids").list.len().alias("count"))
        .filter(pl.col("count") >= threshold)
        .with_columns(pl.concat_list(pl.col(item).cast(pl.String)).alias("items"))
        .select("items", "_tids", "count")
    )
    levels = [singles]
    level = singles
    k = 1
    while k < max_size and level.height > 1:
        joinable = level.with_columns(
            pl.col("items").list.head(k - 1).list.join(_PREFIX_SEP).alias("_prefix"),
            pl.col("items").list.last().alias("_last"),
        )
        right = joinable.select(
            "_prefix",
            pl.col("_last").alias("_last_b"),
            pl.col("_tids").alias("_tids_b"),
        )
        level = (
            joinable.join(right, on="_prefix")
            .filter(pl.col("_last") < pl.col("_last_b"))  # each (k+1)-set generated once, sorted
            .with_columns(
                pl.concat_list("items", "_last_b").alias("items"),
                pl.col("_tids").list.set_intersection("_tids_b").alias("_tids"),
            )
            .with_columns(pl.col("_tids").list.len().alias("count"))
            .filter(pl.col("count") >= threshold)
            .select("items", "_tids", "count")
        )
        if level.height == 0:
            break
        levels.append(level)
        k += 1
    return [
        frame.with_columns((pl.col("count") / n_tx).alias("support")).select(
            "items", "count", "support"
        )
        for frame in levels
    ]


def frequent_itemsets(
    df: pl.DataFrame,
    *,
    transaction: str,
    item: str,
    min_support: float = 0.01,
    max_size: int = 3,
) -> pl.DataFrame:
    """Itemsets of size 1..``max_size`` appearing in at least ``min_support`` of transactions.

    Support = share of transactions containing the whole set. ``max_size`` is a genuine knob
    (raise it for 4+-item bundles); rare items can't form frequent sets, so the ladder prunes
    itself. Returns (``items`` sorted List[str], ``size``, ``count``, ``support``).
    """
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    base, n_tx = _transactions(df, transaction, item)
    levels = _frequent_levels(
        base,
        transaction=transaction,
        item=item,
        n_tx=n_tx,
        min_support=min_support,
        max_size=max_size,
    )
    sized = [frame.with_columns(pl.col("items").list.len().alias("size")) for frame in levels]
    return (
        pl.concat(sized)
        .select("items", "size", "count", "support")
        .sort(["size", "support"], descending=[False, True])
    )


def association_rules(
    df: pl.DataFrame,
    *,
    transaction: str,
    item: str,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
) -> pl.DataFrame:
    """Item→item rules ranked by lift: antecedent, consequent, support, confidence, lift, leverage.

    Confidence = P(consequent | antecedent) — the cross-sell hit rate if you recommend on the
    rule. Lift > 1 = together more often than chance; lift is what separates a real affinity
    from two items that are simply both popular. Leverage is the same gap in absolute share.
    Shares the frequent-set miner with :func:`frequent_itemsets` (singles + pairs only).
    """
    base, n_tx = _transactions(df, transaction, item)
    levels = _frequent_levels(
        base, transaction=transaction, item=item, n_tx=n_tx, min_support=min_support, max_size=2
    )
    if len(levels) < 2 or levels[1].is_empty():
        raise ValueError("no frequent pairs at this min_support — lower it or check the data")

    supports = levels[0].select(
        pl.col("items").list.first().alias("_item"), pl.col("support").alias("_s")
    )
    pairs = levels[1].select(
        pl.col("items").list.get(0).alias("a"),
        pl.col("items").list.get(1).alias("b"),
        "support",
        "count",
    )
    directed = pl.concat(
        [
            pairs.select(
                pl.col("a").alias("antecedent"), pl.col("b").alias("consequent"), "support", "count"
            ),
            pairs.select(
                pl.col("b").alias("antecedent"), pl.col("a").alias("consequent"), "support", "count"
            ),
        ]
    )
    rules = (
        directed.join(supports, left_on="antecedent", right_on="_item")
        .rename({"_s": "support_antecedent"})
        .join(supports, left_on="consequent", right_on="_item")
        .rename({"_s": "support_consequent"})
        .with_columns((pl.col("support") / pl.col("support_antecedent")).alias("confidence"))
        .with_columns(
            (pl.col("confidence") / pl.col("support_consequent")).alias("lift"),
            (pl.col("support") - pl.col("support_antecedent") * pl.col("support_consequent")).alias(
                "leverage"
            ),
        )
        .filter(pl.col("confidence") >= min_confidence)
        .select("antecedent", "consequent", "count", "support", "confidence", "lift", "leverage")
    )
    return rules.sort("lift", descending=True)
