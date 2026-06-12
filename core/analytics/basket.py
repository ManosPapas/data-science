"""Market basket analysis — frequent itemsets and association rules, in pure Polars.

What sells together: support (how often), confidence (P(B|A)), and lift (how much more often
than chance). Drives cross-sell placement, bundling, and the co-purchase graph
(``analytics.graph``). Apriori pruning keeps it tractable: only items above ``min_support`` can
appear in larger sets.
"""

from __future__ import annotations

import polars as pl


def _transactions(df: pl.DataFrame, transaction: str, item: str) -> tuple[pl.DataFrame, int]:
    base = df.select(transaction, item).drop_nulls().unique()
    n_tx = base[transaction].n_unique()
    if n_tx == 0:
        raise ValueError("no transactions found")
    return base, n_tx


def frequent_itemsets(
    df: pl.DataFrame,
    *,
    transaction: str,
    item: str,
    min_support: float = 0.01,
    max_size: int = 3,
) -> pl.DataFrame:
    """Itemsets (size 1-3) appearing in at least ``min_support`` of transactions.

    Support = share of transactions containing the whole set. Raise ``min_support`` if the pair
    join explodes on a long catalogue — rare items can't form frequent sets anyway (apriori).
    """
    if not 1 <= max_size <= 3:
        raise ValueError("max_size must be 1, 2, or 3")
    base, n_tx = _transactions(df, transaction, item)

    singles = (
        base.group_by(item)
        .len()
        .with_columns((pl.col("len") / n_tx).alias("support"))
        .filter(pl.col("support") >= min_support)
    )
    frames = [
        singles.select(
            pl.concat_list(pl.col(item).cast(pl.String)).alias("items"),
            pl.lit(1).alias("size"),
            pl.col("len").alias("count"),
            "support",
        )
    ]

    kept = base.filter(pl.col(item).is_in(singles[item].implode()))
    if max_size >= 2:
        pairs = (
            kept.join(kept, on=transaction, suffix="_b")
            .filter(pl.col(item) < pl.col(f"{item}_b"))
            .group_by(item, f"{item}_b")
            .len()
            .with_columns((pl.col("len") / n_tx).alias("support"))
            .filter(pl.col("support") >= min_support)
        )
        frames.append(
            pairs.select(
                pl.concat_list(
                    pl.col(item).cast(pl.String), pl.col(f"{item}_b").cast(pl.String)
                ).alias("items"),
                pl.lit(2).alias("size"),
                pl.col("len").alias("count"),
                "support",
            )
        )
        if max_size >= 3 and pairs.height > 0:
            # Apriori: a frequent triple's items all sit in some frequent pair.
            pair_items = pl.concat([pairs[item], pairs[f"{item}_b"].rename(item)]).unique()
            narrowed = kept.filter(pl.col(item).is_in(pair_items.implode()))
            triples = (
                narrowed.join(narrowed, on=transaction, suffix="_b")
                .filter(pl.col(item) < pl.col(f"{item}_b"))
                .join(narrowed, on=transaction, suffix="_c")
                .filter(pl.col(f"{item}_b") < pl.col(f"{item}_c"))
                .group_by(item, f"{item}_b", f"{item}_c")
                .len()
                .with_columns((pl.col("len") / n_tx).alias("support"))
                .filter(pl.col("support") >= min_support)
            )
            frames.append(
                triples.select(
                    pl.concat_list(
                        pl.col(item).cast(pl.String),
                        pl.col(f"{item}_b").cast(pl.String),
                        pl.col(f"{item}_c").cast(pl.String),
                    ).alias("items"),
                    pl.lit(3).alias("size"),
                    pl.col("len").alias("count"),
                    "support",
                )
            )
    return pl.concat(frames).sort(["size", "support"], descending=[False, True])


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
    """
    base, n_tx = _transactions(df, transaction, item)
    singles = (
        base.group_by(item)
        .len()
        .with_columns((pl.col("len") / n_tx).alias("support"))
        .filter(pl.col("support") >= min_support)
    )
    kept = base.filter(pl.col(item).is_in(singles[item].implode()))
    pairs = (
        kept.join(kept, on=transaction, suffix="_b")
        .filter(pl.col(item) < pl.col(f"{item}_b"))
        .group_by(item, f"{item}_b")
        .len()
        .with_columns((pl.col("len") / n_tx).alias("support"))
        .filter(pl.col("support") >= min_support)
    )
    if pairs.is_empty():
        raise ValueError("no frequent pairs at this min_support — lower it or check the data")

    supports = singles.select(pl.col(item).alias("_item"), pl.col("support").alias("_s"))
    directed = pl.concat(
        [
            pairs.select(
                pl.col(item).alias("antecedent"),
                pl.col(f"{item}_b").alias("consequent"),
                pl.col("support"),
                pl.col("len").alias("count"),
            ),
            pairs.select(
                pl.col(f"{item}_b").alias("antecedent"),
                pl.col(item).alias("consequent"),
                pl.col("support"),
                pl.col("len").alias("count"),
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
        .select(
            "antecedent",
            "consequent",
            "count",
            "support",
            "confidence",
            "lift",
            "leverage",
        )
    )
    return rules.sort("lift", descending=True)
