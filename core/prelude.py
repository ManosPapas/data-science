"""One-line toolkit for notebooks: ``from core.prelude import *``.

Curated re-exports of the standard kit so the interactive layer imports once. Library modules import
what they need directly — Python caches modules, so that costs nothing and keeps them independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from core.analytics import stats
from core.api.client import get_client, paginate
from core.db.engine import get_engine
from core.db.query import load_sql_file, read_sql, write_sql
from core.io.cache import cached
from core.io.readers import (
    query_files,
    read_csv,
    read_excel,
    read_json,
    read_ndjson,
    read_parquet,
    scan_parquet,
)
from core.io.writers import write_csv, write_excel, write_parquet
from core.utils.memory import memory_report
from core.viz import base, cluster, conceptual, eda, explain, model, timeseries
from core.viz.base import set_theme

__all__ = [
    "base",
    "cached",
    "cluster",
    "conceptual",
    "cs",
    "eda",
    "explain",
    "get_client",
    "get_engine",
    "load_sql_file",
    "memory_report",
    "model",
    "np",
    "paginate",
    "pd",
    "pl",
    "query_files",
    "read_csv",
    "read_excel",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_sql",
    "scan_parquet",
    "set_theme",
    "stats",
    "timeseries",
    "write_csv",
    "write_excel",
    "write_parquet",
    "write_sql",
]
