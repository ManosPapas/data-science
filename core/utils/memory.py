"""DataFrame memory profiling — find the column to downcast or drop.

When a frame is surprisingly large it's almost always one wide string column that wants
``Categorical``, or ``Float64`` where ``Float32`` would do. Measure, don't guess.
"""

from __future__ import annotations

import polars as pl


def frame_size_mb(df: pl.DataFrame) -> float:
    """Estimated in-memory size of the whole frame, in megabytes."""
    return round(float(df.estimated_size("mb")), 2)


def column_sizes_mb(df: pl.DataFrame) -> dict[str, float]:
    """Estimated in-memory size of each column, in megabytes."""
    return {name: round(float(df.select(name).estimated_size("mb")), 3) for name in df.columns}


def memory_report(df: pl.DataFrame) -> str:
    """A human-readable report: total size plus per-column sizes, largest first."""
    by_column = sorted(column_sizes_mb(df).items(), key=lambda item: item[1], reverse=True)
    lines = [f"  {name:<30} {size:>8.3f} MB" for name, size in by_column]
    return f"shape={df.shape}  total={frame_size_mb(df)} MB\n" + "\n".join(lines)
