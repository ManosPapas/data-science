"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-notebooks",
        action="store_true",
        default=False,
        help="also run the slow end-to-end example-notebook tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting for the viz tests
    config.addinivalue_line("markers", "notebook: slow end-to-end example-notebook execution")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-notebooks"):
        return
    skip = pytest.mark.skip(reason="needs --run-notebooks")
    for item in items:
        if "notebook" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG so tests are deterministic."""
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    """Close any matplotlib figures a test opened (only if pyplot was imported)."""
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
