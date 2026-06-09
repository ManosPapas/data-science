"""Parameterized SQL reads/writes returning Polars. SQL files live under ``sql/<workspace>/``."""

from __future__ import annotations

import polars as pl
from sqlalchemy import text

from core.config import ROOT
from core.db.engine import get_engine

SQL_DIR = ROOT / "sql"


def load_sql_file(name: str) -> str:
    """Read a ``.sql`` file by path relative to ``sql/`` (e.g. ``'apo/active_records'``)."""
    path = SQL_DIR / f"{name}.sql"
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def read_sql(
    query: str, conn: str, *, params: dict[str, object] | None = None, from_file: bool = True
) -> pl.DataFrame:
    """Run a parameterized query on the named connection and return a Polars frame.

    ``query`` is a path under ``sql/`` (default) or a raw SQL string when ``from_file=False``.
    Parameters are BOUND (``:name``) — never string-formatted, so this can't be SQL-injected.
    """
    sql = load_sql_file(query) if from_file else query
    with get_engine(conn).connect() as connection:
        return pl.read_database(text(sql), connection, execute_options={"parameters": params or {}})


def write_sql(df: pl.DataFrame, table: str, conn: str, *, if_exists: str = "append") -> None:
    """Write a Polars frame to ``table`` on the named connection (append / replace / fail)."""
    df.write_database(table, get_engine(conn), if_table_exists=if_exists)
