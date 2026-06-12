"""One-line toolkit for notebooks: ``from core.prelude import *``.

Curated re-exports of the standard kit so the interactive layer imports once. Library modules import
what they need directly — Python caches modules, so that costs nothing and keeps them independent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from core import pricing
from core.analytics import causal, experiment, regression, stats
from core.api.client import get_client, paginate
from core.api.graphql import graphql, paginate_graphql
from core.db.engine import get_engine
from core.db.query import load_sql_file, read_sql, write_sql
from core.decision import bandits, optimize
from core.features import clean, geo, period, temporal, text, transform, validate
from core.forecasting import backtest, diagnostics
from core.forecasting.models import make_forecaster
from core.io import catalog
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
    anomaly,
    compare,
    ensemble,
    evaluate,
    imbalance,
    monitor,
    persist,
    preprocess,
    registry,
    segment,
    split,
    train,
    tune,
)
from core.utils import report
from core.utils.memory import memory_report
from core.viz import base, cluster, conceptual, eda, explain, interactive, model, timeseries
from core.viz.base import set_theme

# Notebook ergonomics — runs on import. This prelude is the interactive layer only (library modules
# never import it, so tests/CI still surface real warnings). Silence noisy third-party warnings and
# show every column when inspecting frames.
warnings.filterwarnings("ignore")
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)
pd.set_option("display.max_columns", None)
set_theme()

__all__ = [
    "anomaly",
    "backtest",
    "bandits",
    "base",
    "behaviour",
    "cached",
    "catalog",
    "causal",
    "clean",
    "cluster",
    "compare",
    "conceptual",
    "cs",
    "diagnostics",
    "eda",
    "ensemble",
    "evaluate",
    "experiment",
    "explain",
    "financial",
    "geo",
    "get_client",
    "get_engine",
    "graphql",
    "imbalance",
    "interactive",
    "load_sql_file",
    "make_forecaster",
    "memory_report",
    "model",
    "monitor",
    "np",
    "optimize",
    "paginate",
    "paginate_graphql",
    "pd",
    "period",
    "persist",
    "pl",
    "preprocess",
    "pricing",
    "profit",
    "query_files",
    "read_csv",
    "read_excel",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_sql",
    "registry",
    "regression",
    "report",
    "scan_parquet",
    "segment",
    "set_theme",
    "split",
    "stats",
    "temporal",
    "text",
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
