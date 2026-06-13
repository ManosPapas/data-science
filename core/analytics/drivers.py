"""Driver & root-cause decomposition — *why* did the metric move, in exactly-summing parts.

The diagnostic layer: every function splits a headline change into named contributions that add
up to the total (no unexplained residual), so the bridge slide and the data agree. Decomposition
is accounting, not causation — it says *where* the change sits, the causal tools say *what made
it happen*.
"""

from __future__ import annotations

import polars as pl


def change_decomposition(
    current: pl.DataFrame, baseline: pl.DataFrame, *, value: str, by: str
) -> pl.DataFrame:
    """Which segments drove the change in a metric's total between two periods?

    Per segment: baseline, current, change, and its share of the total move. Contributions sum
    exactly to the headline change; segments present in only one period count in full (entries
    and exits are often the real story). Sorted by |change| — the root-cause shortlist.
    """
    base = baseline.group_by(by).agg(pl.col(value).sum().alias("baseline"))
    cur = current.group_by(by).agg(pl.col(value).sum().alias("current"))
    joined = (
        base.join(cur, on=by, how="full", coalesce=True)
        .with_columns(pl.col("baseline").fill_null(0.0), pl.col("current").fill_null(0.0))
        .with_columns((pl.col("current") - pl.col("baseline")).alias("change"))
    )
    total_change = joined["change"].sum()
    return joined.with_columns(
        pl.when(total_change != 0)
        .then(pl.col("change") / total_change)
        .otherwise(None)
        .alias("share_of_change")
    ).sort(pl.col("change").abs(), descending=True)


def price_volume_mix(
    current: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    price: str,
    volume: str,
    by: str,
) -> pl.DataFrame:
    """Revenue bridge: split ΔRevenue into price, volume, and mix effects per segment.

    - ``price_effect``: charging differently for the same units (realized on current volume);
    - ``volume_effect``: the market growing/shrinking at the old mix and old prices;
    - ``mix_effect``: share shifting between cheap and expensive segments at old prices.

    The three sum *exactly* to the revenue change. A revenue rise that is all mix (customers
    trading up) needs a different response than a price rise — this is also the decomposition
    that separates "our pricing worked" from "the mix flattered us".
    """
    base = baseline.group_by(by).agg(
        pl.col(volume).sum().alias("volume_0"),
        (pl.col(price) * pl.col(volume)).sum().alias("revenue_0"),
    )
    cur = current.group_by(by).agg(
        pl.col(volume).sum().alias("volume_1"),
        (pl.col(price) * pl.col(volume)).sum().alias("revenue_1"),
    )
    joined = (
        base.join(cur, on=by, how="full", coalesce=True)
        .with_columns(
            pl.col("volume_0").fill_null(0.0),
            pl.col("revenue_0").fill_null(0.0),
            pl.col("volume_1").fill_null(0.0),
            pl.col("revenue_1").fill_null(0.0),
        )
        .with_columns(
            # Per-segment average prices; a segment absent in one period inherits the other
            # period's price so its whole contribution lands in volume/mix, not price.
            (pl.col("revenue_0") / pl.col("volume_0")).alias("price_0"),
            (pl.col("revenue_1") / pl.col("volume_1")).alias("price_1"),
        )
        .with_columns(
            # 0/0 → NaN and revenue/0 → ±inf both mean "no real per-unit price here"; null them
            # so the inheriting fill_null below carries the other period's price (whole
            # contribution lands in volume/mix), keeping the bridge finite and exactly-summing.
            pl.when(pl.col("price_0").is_finite()).then(pl.col("price_0")).alias("price_0"),
            pl.when(pl.col("price_1").is_finite()).then(pl.col("price_1")).alias("price_1"),
        )
        .with_columns(
            pl.col("price_0").fill_null(pl.col("price_1")),
            pl.col("price_1").fill_null(pl.col("price_0")),
        )
    )
    total_volume_0 = float(joined["volume_0"].sum() or 0.0)
    total_volume_1 = float(joined["volume_1"].sum() or 0.0)
    if total_volume_0 == 0 or total_volume_1 == 0:
        raise ValueError("baseline or current volume is zero — the bridge is undefined")
    result = (
        joined.with_columns(
            (pl.col("volume_0") / total_volume_0).alias("share_0"),
            (pl.col("volume_1") / total_volume_1).alias("share_1"),
        )
        .with_columns(
            ((pl.col("price_1") - pl.col("price_0")) * pl.col("volume_1")).alias("price_effect"),
            ((total_volume_1 - total_volume_0) * pl.col("share_0") * pl.col("price_0")).alias(
                "volume_effect"
            ),
            (total_volume_1 * (pl.col("share_1") - pl.col("share_0")) * pl.col("price_0")).alias(
                "mix_effect"
            ),
        )
        .with_columns(
            (pl.col("price_effect") + pl.col("volume_effect") + pl.col("mix_effect")).alias(
                "total_effect"
            )
        )
        .select(
            by,
            "revenue_0",
            "revenue_1",
            "price_effect",
            "volume_effect",
            "mix_effect",
            "total_effect",
        )
    )
    return result.sort(pl.col("total_effect").abs(), descending=True)


def revenue_leakage(
    df: pl.DataFrame, *, expected: str, actual: str, by: str | None = None
) -> pl.DataFrame:
    """Where realized revenue falls short of entitled revenue — ranked leak detection.

    ``expected`` is what the books say you should have collected (list/contract price, approved
    discount); ``actual`` is what landed. Positive leakage = systematic under-realization
    (unapproved discounting, billing gaps, fee waivers); a negative number is over-realization.
    Group ``by`` rep/segment/product to find *where* the margin quietly leaks.
    """
    if by is not None:
        grouped = df.group_by(by).agg(
            pl.col(expected).sum().alias("expected"), pl.col(actual).sum().alias("actual")
        )
    else:
        grouped = df.select(
            pl.col(expected).sum().alias("expected"), pl.col(actual).sum().alias("actual")
        )
    return (
        grouped.with_columns((pl.col("expected") - pl.col("actual")).alias("leakage"))
        .with_columns(
            pl.when(pl.col("expected") != 0)
            .then(pl.col("leakage") / pl.col("expected"))
            .otherwise(None)
            .alias("leakage_rate")
        )
        .sort("leakage", descending=True)
    )
