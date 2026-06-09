"""Readers — scan/read tabular data with memory-aware defaults.

Prefer the lazy :func:`scan_parquet` so filters and column selection push down and only the bytes
you need leave disk. Use eager reads for small data. ``read_excel`` needs the ``excel`` extra.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


def scan_parquet(path: str | Path) -> pl.LazyFrame:
    """Lazily scan Parquet (a file, glob, or directory). Nothing is read until ``.collect()``."""
    return pl.scan_parquet(str(path))


def read_parquet(path: str | Path) -> pl.DataFrame:
    """Eagerly read Parquet into memory. Prefer ``scan_parquet`` for large data."""
    return pl.read_parquet(str(path))


def read_csv(path: str | Path, *, try_parse_dates: bool = True) -> pl.DataFrame:
    """Read a CSV into a Polars frame, parsing date-like columns by default."""
    return pl.read_csv(str(path), try_parse_dates=try_parse_dates)


def read_json(path: str | Path) -> pl.DataFrame:
    """Read a JSON file (array of records) into a Polars frame."""
    return pl.read_json(str(path))


def read_ndjson(path: str | Path) -> pl.DataFrame:
    """Read newline-delimited JSON into a Polars frame."""
    return pl.read_ndjson(str(path))


def read_excel(path: str | Path, *, sheet: int | str = 1) -> pl.DataFrame:
    """Read one sheet of an .xlsx file (1-based index or name). Needs the ``excel`` extra."""
    if isinstance(sheet, int):
        return pl.read_excel(str(path), sheet_id=sheet)
    return pl.read_excel(str(path), sheet_name=sheet)


def query_files(sql: str) -> pl.DataFrame:
    """Run a DuckDB SQL query over files and return a Polars frame.

    Example: ``query_files("SELECT * FROM 'data/raw/*.parquet' WHERE region = 'EU'")``.
    DuckDB streams over the files out-of-core, so this stays bounded even when inputs exceed RAM.
    """
    result: pl.DataFrame = duckdb.sql(sql).pl()
    return result
