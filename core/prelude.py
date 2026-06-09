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
from core.io.readers import query_files, read_csv, read_parquet, scan_parquet
from core.utils.memory import memory_report
from core.viz import base, cluster, conceptual, eda, explain, model, timeseries
from core.viz.base import set_theme

__all__ = [
    "base",
    "cluster",
    "conceptual",
    "cs",
    "eda",
    "explain",
    "memory_report",
    "model",
    "np",
    "pd",
    "pl",
    "query_files",
    "read_csv",
    "read_parquet",
    "scan_parquet",
    "set_theme",
    "stats",
    "timeseries",
]
