"""Writers — persist Polars frames. Parquet is the default for anything that sticks around."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    """Write a frame to Parquet, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)


def write_csv(df: pl.DataFrame, path: str | Path) -> None:
    """Write a frame to CSV (prefer Parquet for anything that persists)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(target)


def write_excel(df: pl.DataFrame, path: str | Path, *, worksheet: str = "Sheet1") -> None:
    """Write a frame to .xlsx (needs the ``excel`` extra)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(target, worksheet=worksheet)
