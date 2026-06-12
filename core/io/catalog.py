"""Typed source catalog — register named datasets once, load them with pinned schemas anywhere.

Operationalizes the "one typed loader per source" rule: a ``Source`` pins where the data lives and
the dtypes it must satisfy; ``load`` reads (or lazily scans) and enforces the pin, failing fast on
a missing column or a value that no longer casts. Register sources once (e.g. in a small
``sources.py`` per workspace, or a notebook's first cell) and every consumer loads by name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl


@dataclass(frozen=True)
class Source:
    """A registered dataset: where it lives plus the schema it must satisfy."""

    name: str
    path: Path
    schema: Mapping[str, Any] | None = None  # column -> polars dtype
    format: str = "parquet"  # parquet | csv


_REGISTRY: dict[str, Source] = {}


def register(
    name: str,
    path: str | Path,
    *,
    schema: Mapping[str, Any] | None = None,
    format: str | None = None,
) -> Source:
    """Register (or replace) a named source; format inferred from the suffix when omitted."""
    target = Path(path)
    fmt = format or ("csv" if target.suffix.lower() == ".csv" else "parquet")
    source = Source(name, target, schema, fmt)
    _REGISTRY[name] = source
    return source


def sources() -> list[str]:
    """Names of all registered sources."""
    return sorted(_REGISTRY)


def describe(name: str) -> Source:
    """The registered ``Source`` for ``name`` (raises with the available names)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}. registered: {', '.join(sources()) or '(none)'}")
    return _REGISTRY[name]


def load(
    name: str, *, lazy: bool = False, columns: Sequence[str] | None = None
) -> pl.DataFrame | pl.LazyFrame:
    """Load a registered source, enforcing its schema pin.

    ``lazy=True`` returns a LazyFrame (the cast is pushed into the scan); eager loads also check
    that every pinned column is present before casting. The cast is strict — a value that no
    longer fits the pinned dtype raises instead of silently turning into null.
    """
    source = describe(name)
    frame: pl.DataFrame | pl.LazyFrame
    if source.format == "csv":
        frame = pl.scan_csv(source.path) if lazy else pl.read_csv(source.path, try_parse_dates=True)
    else:
        frame = pl.scan_parquet(source.path) if lazy else pl.read_parquet(source.path)
    if columns is not None:
        frame = frame.select(list(columns))
    if source.schema:
        pinned = {k: v for k, v in source.schema.items() if columns is None or k in columns}
        if isinstance(frame, pl.DataFrame):
            missing = [col for col in pinned if col not in frame.columns]
            if missing:
                raise ValueError(f"source {name!r} is missing pinned columns: {missing}")
        frame = frame.cast(cast(Any, pinned))
    return frame
