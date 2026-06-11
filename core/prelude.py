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
from core.api.graphql import graphql, paginate_graphql
from core.db.engine import get_engine
from core.db.query import load_sql_file, read_sql, write_sql
from core.decision import bandits
from core.features import clean, period, temporal, transform, validate
from core.forecasting import backtest
from core.forecasting.models import make_forecaster
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
from core.kpi import behaviour, financial, profit
from core.modeling import (
    compare,
    ensemble,
    evaluate,
    imbalance,
    persist,
    preprocess,
    registry,
    split,
    train,
    tune,
)
from core.utils.memory import memory_report
from core.viz import base, cluster, conceptual, eda, explain, model, timeseries
from core.viz.base import set_theme

__all__ = [
    "backtest",
    "bandits",
    "base",
    "behaviour",
    "cached",
    "clean",
    "cluster",
    "compare",
    "conceptual",
    "cs",
    "eda",
    "ensemble",
    "evaluate",
    "explain",
    "financial",
    "get_client",
    "get_engine",
    "graphql",
    "imbalance",
    "load_sql_file",
    "make_forecaster",
    "memory_report",
    "model",
    "np",
    "paginate",
    "paginate_graphql",
    "pd",
    "period",
    "persist",
    "pl",
    "preprocess",
    "profit",
    "query_files",
    "read_csv",
    "read_excel",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_sql",
    "registry",
    "scan_parquet",
    "set_theme",
    "split",
    "stats",
    "temporal",
    "timeseries",
    "train",
    "transform",
    "tune",
    "validate",
    "write_csv",
    "write_excel",
    "write_parquet",
    "write_sql",
]
