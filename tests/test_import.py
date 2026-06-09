"""Smoke test — the package imports and settings load."""

from __future__ import annotations

import core
from core.config import get_settings


def test_package_version() -> None:
    assert isinstance(core.__version__, str)


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.environment  # defaults to "dev"
    assert "paths" in settings.conf
