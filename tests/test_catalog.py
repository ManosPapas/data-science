"""Tests for the typed source catalog."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from core.io import catalog


def test_register_load_roundtrip_with_pinned_schema(tmp_path: Path) -> None:
    path = tmp_path / "trips.parquet"
    pl.DataFrame({"id": [1, 2], "value": [1.5, 2.5]}).write_parquet(path)
    catalog.register("trips", path, schema={"id": pl.Int32, "value": pl.Float32})
    out = catalog.load("trips")
    assert isinstance(out, pl.DataFrame)
    assert out.schema["id"] == pl.Int32  # the pin is enforced, not the file's Int64
    lazy = catalog.load("trips", lazy=True)
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().height == 2


def test_load_missing_pinned_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "orders.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    catalog.register("orders-broken", path, schema={"x": pl.Int64, "gone": pl.Int64})
    with pytest.raises(ValueError, match="gone"):
        catalog.load("orders-broken")


def test_unknown_source_lists_registered() -> None:
    with pytest.raises(KeyError, match="unknown source"):
        catalog.describe("never-registered")


def test_csv_format_inferred(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    pl.DataFrame({"a": [1, 2]}).write_csv(path)
    source = catalog.register("raw-csv", path)
    assert source.format == "csv"
    loaded = catalog.load("raw-csv")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.height == 2
