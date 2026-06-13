"""Sequential risk re-scoring — update a prediction as an entity moves through checkpoints.

A model trained at one decision point can be re-run at every later checkpoint as more state
arrives: the same fitted estimator, scored on a sequence of progressively richer feature
snapshots, yields a risk *trajectory* per entity. Generalized from milestone risk scoring — the
shape fits loan repayment milestones, return-risk after purchase, patient-outcome visits, any
checkpointed process. Each snapshot must be leak-free (only state known at that checkpoint).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from core.modeling.train import score


def rescore_sequence(
    model: Any,
    snapshots: Mapping[str, pl.DataFrame],
    *,
    feature_columns: list[str],
    id_column: str | None = None,
    positive_class: int = 1,
) -> pl.DataFrame:
    """Score ``model`` on each checkpoint snapshot; return the long risk trajectory.

    ``snapshots`` maps a checkpoint label → that checkpoint's feature frame (same entities, more
    information as you go). Returns (``checkpoint``, [``id_column``], ``risk``) — the per-entity
    path of the score over time. Insertion order of ``snapshots`` is the checkpoint order. Pair
    with :func:`core.operational.alerts.generate_alerts` to turn the trajectory into actions, or
    chart the rising/falling risk per entity. ``feature_columns`` pins the model's inputs so an
    extra column in a later snapshot can't silently reorder the matrix.
    """
    if not snapshots:
        raise ValueError("no snapshots to score")
    frames = []
    for checkpoint, snapshot in snapshots.items():
        missing = [c for c in feature_columns if c not in snapshot.columns]
        if missing:
            raise ValueError(f"snapshot {checkpoint!r} is missing feature columns: {missing}")
        risk = score(model, snapshot.select(feature_columns), positive_class=positive_class)
        data: dict[str, Any] = {"checkpoint": [checkpoint] * snapshot.height}
        if id_column is not None:
            data[id_column] = snapshot[id_column]
        data["risk"] = risk
        frames.append(pl.DataFrame(data))
    # vertical_relaxed supertypes an id_column whose dtype drifts across checkpoints (Int32/Int64)
    return pl.concat(frames, how="vertical_relaxed")
