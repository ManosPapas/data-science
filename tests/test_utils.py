"""Tests for utility helpers."""

from __future__ import annotations

import polars as pl

from core.utils import logging as log
from core.utils.memory import frame_size_mb


def test_get_logger_name() -> None:
    assert log.get_logger("core.test").name == "core.test"


def test_frame_size_mb() -> None:
    assert frame_size_mb(pl.DataFrame({"a": [1, 2, 3]})) >= 0.0
