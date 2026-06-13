"""Data / feed readiness — "do we even have the data to score this?"

Before any model ships against a live feed (events, milestones, custody checkpoints), the first
question is coverage: which expected signals actually arrive, for what share of entities, and who
is missing the ones a score depends on. Generalized from operational milestone-coverage
diagnostics — works for any process with named checkpoints (delivery scans, funnel steps,
clinical visits, manufacturing stations).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def feed_readiness(
    df: pl.DataFrame,
    *,
    entity: str,
    milestone: str,
    expected: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> pl.DataFrame:
    """Per-milestone coverage across entities: present count, share, and recency.

    ``df`` is a long event log (one row per entity-milestone occurrence). For each milestone:
    ``entities`` with it, ``coverage`` (share of all entities), and — if ``timestamp`` is given —
    the latest occurrence (a stale feed is as bad as a missing one). Pass ``expected`` to surface
    milestones that *should* arrive but never do (coverage 0) instead of silently omitting them.
    The go/no-go gate: a model that needs a 5%-coverage milestone can't be scored live yet.
    """
    clean = df.drop_nulls([entity, milestone])
    total = clean[entity].n_unique()
    if total == 0:
        raise ValueError("no entities found")
    aggs = [pl.col(entity).n_unique().alias("entities")]
    if timestamp is not None:
        aggs.append(pl.col(timestamp).max().alias("latest"))
    coverage = (
        clean.group_by(milestone)
        .agg(aggs)
        .with_columns((pl.col("entities") / total).alias("coverage"))
    )
    if expected is not None:
        seen = set(coverage[milestone].to_list())
        missing = [m for m in expected if m not in seen]
        if missing:
            filler = pl.DataFrame({milestone: missing}).with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("entities"),
                pl.lit(0.0).alias("coverage"),
                *([pl.lit(None).alias("latest")] if timestamp is not None else []),
            )
            coverage = pl.concat([coverage, filler], how="diagonal")
    return coverage.sort("coverage", descending=True)


def entities_missing(
    df: pl.DataFrame, *, entity: str, milestone: str, required: Sequence[str]
) -> pl.DataFrame:
    """Entities lacking one or more ``required`` milestones — the rows you cannot score yet.

    Returns one row per entity short of a required milestone, with the ``missing`` list. Feed it
    to triage (chase the feed, hold the score) before a model runs on incomplete custody.
    """
    clean = df.drop_nulls([entity, milestone])
    have = clean.group_by(entity).agg(pl.col(milestone).unique().alias("_have"))
    required_list = list(required)
    return (
        have.with_columns(
            pl.lit(required_list).list.set_difference(pl.col("_have")).alias("missing")
        )
        .filter(pl.col("missing").list.len() > 0)
        .select(entity, "missing")
        .sort(entity)
    )
