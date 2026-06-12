"""Pooled SQLAlchemy engines built from the named connections in ``config``.

Targets SQL Server via ODBC (matching the classic SERVER/USER_NAME/PASSWORD/DATABASE/DRIVER block).
Other dialects can be supported by extending the URL built here.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine

from core.config import get_connection


@lru_cache
def get_engine(name: str) -> Engine:
    """Pooled SQLAlchemy engine for the named connection (cached per name)."""
    conn = get_connection(name)
    password = conn.password.get_secret_value() if conn.password else ""
    odbc = (
        f"DRIVER={{{conn.driver}}};SERVER={conn.server};DATABASE={conn.database};"
        f"UID={conn.user_name};PWD={password}"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}", pool_pre_ping=True)
